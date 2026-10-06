# Streaming wakeword float32 model

The application stores only
`str_ww_ref_model_float32.tflite`, converted from the MLPerf Tiny v1.4 H5
reference model at
`.envs/tiny-v1.4/benchmark/training/streaming_wakeword/trained_models/str_ww_ref_model.h5`
by `scripts/convert_sww_model.py`. The H5 and conversion intermediates remain
outside this application directory.

The output is 191,428 bytes with SHA-256
`c735ab47248df7648d9cb4397c0e7d161fe2e88ede17ad900f34a4163d89b267`. Its
FlatBuffer has one input `serving_default_input_1:0`, float32 `[1, 30, 1, 40]`,
and one output `StatefulPartitionedCall:0`, float32 `[1, 3]`. The 11 operators
are four depthwise convolutions, four convolutions, reshape, fully connected,
and softmax. Weights and activations are float32; the reshape shape constant
is int32.

Conversion provenance:

- H5 SHA-256: `b0f267a8ba0bcb911c1098229c32fac21996e4191c9c60d1fda80adaa70a8add`
- upstream `quantize.py` SHA-256: `6303e820a13ce6d50ea26f2e6d19ee3cbbfac2fae99cb520c0737afc578742f2`
- conversion environment: `.envs/sww-env`, Python 3.11, exact supplied `requirements.txt`
