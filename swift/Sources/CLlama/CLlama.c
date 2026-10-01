//
//  CLlama.c
//  CLlama
//
//  A header-only C target confuses SwiftPM's linker step (it expects an
//  object file that never gets produced), so this translation unit exists
//  purely to give the target something to compile. It carries no code.
//

#include "CLlama.h"
