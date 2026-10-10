"""Real production geometry and independent nonzero layout expectations."""
import json
from pathlib import Path
import numpy as np
import pytest
from vta.environment import Environment


def environment(bi, bo, batch=0):
    cfg = json.loads((Path(__file__).resolve().parents[3] / 'config/vta_64mac.json').read_text())
    cfg.update(TARGET="sim", LOG_BLOCK_IN=bi, LOG_BLOCK_OUT=bo, LOG_BATCH=batch)
    return Environment(cfg)


@pytest.mark.parametrize('bi,bo,depth,subdepth,banks,bankdepth', [
    (3,3,1024,1024,1,1024), (3,4,1024,1024,2,512), (4,3,512,1024,2,512)])
def test_production_geometry_and_abi(bi,bo,depth,subdepth,banks,bankdepth):
    env = environment(bi, bo)
    assert (env.BLOCK_IN,env.BLOCK_OUT) == (1<<bi,1<<bo)
    assert env.INP_BUFF_DEPTH == depth
    assert env.INP_SUBVECTOR_DEPTH == subdepth
    assert env.INP_BANK_BITS == 64
    assert env.INP_BANK_COUNT == banks
    assert env.INP_BANK_DEPTH == bankdepth
    assert env.INP_PARALLEL_BITS == 64*banks
    from vta.environment import pkg_config
    assert len({environment(i,o).BITSTREAM for i,o in [(3,3),(3,4),(4,3)]}) == 3


@pytest.mark.parametrize('bi,bo', [(3,3),(3,4),(4,3)])
def test_input_spatial_channel_vectors_and_real_halves(bi,bo):
    from vta.top.dwc_layout import pack_dwc_input
    env = environment(bi,bo)
    data = np.arange(1,65,dtype='int8').reshape(1,16,2,2)
    packed = pack_dwc_input(data,env)
    assert packed.data.flags.c_contiguous
    np.testing.assert_array_equal(packed.unpack(),data)
    # First spatial point, all 16 channels; every element is actual input.
    vectors = packed.data.reshape(-1,env.BATCH,env.BLOCK_IN)
    np.testing.assert_array_equal(vectors[:16//env.BLOCK_IN].reshape(-1), np.arange(1,65,4))
    assert packed.source_index(1,0,0) == 4
    if bi==4:
        assert packed.source_index(0,0,1) == 1
        assert packed.bank_address(0,0,0) == (0,0)
        assert packed.bank_address(0,0,1) == (1,0)
    elif bo==4:
        assert packed.bank_address(0) == (0,0)
        assert packed.bank_address(1) == (1,0)
        with pytest.raises(ValueError):
            packed.source_index(0,0,0,base_vector=1)


@pytest.mark.parametrize('bi,bo', [(3,3),(3,4),(4,3)])
def test_weight_actual_row_major_taps_and_tail(bi,bo):
    from vta.top.dwc_layout import pack_dwc_weight
    env=environment(bi,bo)
    kernel=(np.arange(16*9)%101-50).astype('int8').reshape(16,3,3)
    packed=pack_dwc_weight(kernel,env)
    assert packed.shape==(16//env.BLOCK_OUT,(9+env.BLOCK_IN-1)//env.BLOCK_IN,env.BLOCK_OUT,env.BLOCK_IN)
    for c in range(16):
        row=packed[c//env.BLOCK_OUT,:,c%env.BLOCK_OUT,:].reshape(-1)
        np.testing.assert_array_equal(row[:9],kernel[c].reshape(-1))
        assert not np.any(row[9:])


def test_input_batch_mapping_and_incomplete_channels_rejected():
    from vta.top.dwc_layout import pack_dwc_input
    env=environment(3,4,1)
    data=np.arange(1,65,dtype='int8').reshape(2,16,1,2)
    packed=pack_dwc_input(data,env)
    np.testing.assert_array_equal(packed.unpack(),data)
    assert packed.bank_address(1,batch=1)==(3,0)
    with pytest.raises(ValueError):
        pack_dwc_input(data[:,:8],env)


def test_production_normalization_fingerprint_and_instruction_capacity():
    import importlib.util
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('geometry',root/'config/vta_config.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fingerprints=set()
    for bi,bo,idepth,wdepth,adepth in [(3,3,1024,256,1024),(3,4,1024,128,512),(4,3,512,128,1024)]:
        cfg=json.loads((root/'config/vta_64mac.json').read_text())
        cfg.update(LOG_BLOCK_IN=bi,LOG_BLOCK_OUT=bo)
        props=module.normalized_chisel_properties(cfg)
        assert (props['BLOCK_IN'],props['BLOCK_OUT'])==(1<<bi,1<<bo)
        assert (props['INP_MEM_DEPTH'],props['WGT_MEM_DEPTH'],props['ACC_MEM_DEPTH'])==(idepth,wdepth,adepth)
        fingerprints.add(module.abi_fingerprint(module.abi_definitions(cfg)))
    assert len(fingerprints)==3
    cfg.update(LOG_INP_BUFF_SIZE=20)
    with pytest.raises(ValueError,match='128-bit|32-bit'):
        module.normalized_chisel_properties(cfg)
