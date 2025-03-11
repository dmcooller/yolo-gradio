from enum import Enum
import logging
from torch.cuda import is_available as cudaIsAvailable
from torch.backends.mps import is_available as mpsIsAvailable

logger = logging.getLogger(__name__)

class DeviceType(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"

def get_device_type(backend_type: DeviceType | str) -> str:
    """Determines the backend type."""

    match backend_type:
        case DeviceType.AUTO:
            if cudaIsAvailable():
                backend = 'cuda'
            elif mpsIsAvailable():
                backend = 'mps'
            else:
                backend = 'cpu'
        case DeviceType.CUDA:
            if cudaIsAvailable():
                backend = 'cuda'
            else:
                raise RuntimeError('CUDA is not available')
        case DeviceType.CPU:
            backend = 'cpu'
        case DeviceType.MPS:
            if mpsIsAvailable():
                backend = 'mps'
            else:
                raise RuntimeError('MPS is not available')
    logger.info('Using backend: %s', backend)
    return backend