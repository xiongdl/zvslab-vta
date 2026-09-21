<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# VTA Geometry Configuration

The canonical shared geometry file is `vta/config/vta_64mac.json`. It is used
for both simulator backends and contains geometry/ABI fields only; it does not
select a simulator. The requested geometry includes:

```text
LOG_BLOCK=3
LOG_UOP_BUFF_SIZE=12
LOG_INP_BUFF_SIZE=13
LOG_WGT_BUFF_SIZE=14
LOG_ACC_BUFF_SIZE=15
```

Select the backend separately with `VTA_BACKEND=fsim` or `VTA_BACKEND=tsim`,
and pass the same absolute config path to the build interface:

```bash
bash scripts/build_vta_lib.sh \
  --config "$PWD/vta/config/vta_64mac.json" \
  --backend fsim

VTA_CONFIG_FILE="$PWD/vta/config/vta_64mac.json" VTA_BACKEND=tsim \
  PYTHONPATH="$PWD/tvm/python:$PWD/vta/python" \
  ./.envs/tvm-vta-env/bin/python -c \
  'import vta; print(vta.get_env().LOG_BLOCK)'
```

`VTA_BACKEND` is not part of the geometry ABI fingerprint, so FSIM and TSIM
artifacts built from this file must agree on geometry. The old
`TARGET=sim`/`TARGET=tsim` configuration selectors are rejected with a
migration error; replace them with the explicit backend environment variable.
The old `--target libvta_*` build option is rejected; use
`--config ABS_PATH --backend fsim|tsim|all`.

FPGA backend names such as `pynq` and `zcu104` remain deferred. This config
contract does not claim those backends are implemented.
