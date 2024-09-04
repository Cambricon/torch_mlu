# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
class TC_Allowlist_Meta(type):
    # Enable grammar sugar as 'v in TC_Allowlist'.
    def __contains__(cls, item):
        return cls.__contains__(item)


class TC_Allowlist(metaclass=TC_Allowlist_Meta):
    # It's NONE for MLU
    allowlist = []

    @classmethod
    def __contains__(cls, item):
        # If kernel name contains substring equal to any one in allowlist, then it uses tensor core.
        for pattern in cls.allowlist:
            if pattern in item:
                return True
        return False


class TC_OP_Allowlist(metaclass=TC_Allowlist_Meta):
    # It's NONE for MLU
    allowlist = []

    @classmethod
    def __contains__(cls, item):
        # If operator name equals to any one in allowlist, then it is tensor core eligible.
        return item in cls.allowlist
