"""Conservative RTX 4060 MM-IMDb reproduction launcher (micro-batch 1)."""

import reproduce_mmimdb


reproduce_mmimdb.PROFILE_NAME = "dcp_mmimdb_reproduction_bs1_seed0"
reproduce_mmimdb.MICRO_BATCH = 1


if __name__ == "__main__":
    reproduce_mmimdb.main()
