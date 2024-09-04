#pragma once

struct AmpUnscaleChunk256 {
  char* input[256];
  uint64_t input_numel[256];
};

struct AmpUnscaleChunk512 {
  char* input[512];
  uint64_t input_numel[512];
};

struct AmpUnscaleChunk1024 {
  char* input[1024];
  uint64_t input_numel[1024];
};
