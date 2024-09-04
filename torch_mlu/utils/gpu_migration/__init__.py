from .migration import apply_monkey_patches
from .env_migration import unsupport_env_check

apply_monkey_patches()
unsupport_env_check()
