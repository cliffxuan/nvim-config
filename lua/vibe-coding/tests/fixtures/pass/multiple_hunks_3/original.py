from loguru import logger

from app.main import store
from app.utils.dailer import get_mixtures


def main():
    with store.bypass_cache():
        store.get_platform_shirts("dailer", bypass_cache=True)
        for mixture, data in get_mixtures("dailer").items():
            if data["state"].upper() != "LIVE":
                logger.info(
                    f"Skipping dailer mixture={mixture} with state={data['state']}"
                )
                continue
            try:
                logger.info(f"Start populating cache for dailer mixture={mixture}")
                store.dailer_get_sync_policies(mixture)
                store.dailer_get_target_policies(mixture)
                store.dailer_get_running_jobs(mixture)
                logger.info(f"Finish populating cache for dailer mixture={mixture}")
            except Exception:
                logger.exception(f"Error populating cache for dailer mixture={mixture}")
        store.get_platform_shirts("fast", bypass_cache=True)
        for mixture, data in get_mixtures("fast").items():
            if data["state"].upper() != "LIVE":
                logger.info(
                    f"Skipping fast mixture={mixture} with state={data['state']}"
                )
                continue
            try:
                logger.info(f"populate cache for fast mixture={mixture}")
                logger.info(f"fast_get_version: {store.fast_get_version(mixture)}")
                store.fast_list_protected_paths(mixture)
                store.fast_list_protection_policies(mixture)
                store.fast_list_replication_streams(mixture)
                logger.info(f"Finish populating cache for fast mixture={mixture}")
            except Exception:
                logger.exception(f"Error populating cache for fast mixture={mixture}")


if __name__ == "__main__":
    main()
