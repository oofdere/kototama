use crate::Token;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;

pub trait Sampler {
    fn name(&self) -> &'static str {
        let full_name = std::any::type_name::<Self>();
        full_name.split("::").last().unwrap_or(full_name)
    }

    fn apply(&self, logits: &[f32]) -> Vec<f32> {
        let mut logits = logits.to_vec();
        self.apply_mut(&mut logits);
        logits
    }

    fn apply_mut(&self, logits: &mut [f32]);

    fn sample(&self, logits: &[f32]) -> Option<Token> {
        let mut logits = logits.to_vec();
        self.apply_mut(&mut logits);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id as i32)
    }

    fn sample_mut(&self, logits: &mut [f32]) -> Option<Token> {
        self.apply_mut(logits);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(id, _)| id as i32)
    }
}

pub struct Temperature{
    pub temp: f32,
}

impl Temperature {
    pub fn new(temp: f32) -> Self {
        Self { temp }
    }
}

impl Sampler for Temperature {
    fn apply_mut(&self, logits: &mut [f32]) {
        if self.temp == 0.0 {
            return;
        }

        let inv_temp = 1.0 / self.temp;

        for logit in logits.iter_mut() {
            *logit *= inv_temp;
        }
    }
}

pub struct MinP {
    pub p: f32,
    pub min_keep: usize,
}

impl MinP {
    pub fn new(p: f32, min_keep: usize) -> Self {
        Self { p, min_keep }
    }
}

impl Sampler for MinP {
    fn apply_mut(&self, logits: &mut [f32]) {
        if logits.len() <= self.min_keep {
            return;
        }

        let Some(&logit_max) = logits.iter().max_by(|a, b| a.total_cmp(b)) else {
            return;
        };
        let mut thresh = logit_max + self.p.ln();

        if self.min_keep > 0 {
            let mut copy: Vec<f32> = logits.to_vec();
            copy.select_nth_unstable_by(self.min_keep - 1, |a, b| b.total_cmp(a));
            thresh = thresh.min(copy[self.min_keep - 1]);
        }

        for logit in logits.iter_mut() {
            if *logit < thresh {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

pub struct Dist {
    rng: RefCell<StdRng>,
}

impl Dist {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl Sampler for Dist {
    fn apply_mut(&self, _logits: &mut [f32]) {}

    fn sample(&self, logits: &[f32]) -> Option<Token> {
        if logits.is_empty() {
            return None;
        }
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - m).exp()).collect();
        let sum: f32 = exps.iter().copied().sum();
        let r = self.rng.borrow_mut().random::<f32>() * sum;
        let mut acc = 0.0;
        for (id, &e) in exps.iter().enumerate() {
            acc += e;
            if acc >= r {
                return Some(id as i32);
            }
        }
        Some((exps.len() - 1) as i32)
    }
}
