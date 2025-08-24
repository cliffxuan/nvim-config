import time
from loguru import logger

from app.main import store
from app.utils.dailer import get_mixtures


def main():
    start_time = time.time()
    logger.info("Starting cache population")
    
    with store.bypass_cache():
        store.get_platform_shirts("dailer", bypass_cache=True)
        for mixture, data in get_mixtures("dailer").items():
            if data["state"].upper() != "LIVE":
                logger.info(
                    f"Skipping dailer mixture={mixture} with state={data['state']}"
                )
                continue
            try:
                mixture_start = time.time()
                logger.info(f"Start populating cache for dailer mixture={mixture}")
                store.dailer_get_sync_policies(mixture)
                store.dailer_get_target_policies(mixture)
                store.dailer_get_running_jobs(mixture)
                mixture_duration = time.time() - mixture_start
                logger.info(f"Finish populating cache for dailer mixture={mixture} in {mixture_duration:.2f}s")
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
                mixture_start = time.time()
                logger.info(f"populate cache for fast mixture={mixture}")
                logger.info(f"fast_get_version: {store.fast_get_version(mixture)}")
                store.fast_list_protected_paths(mixture)
                store.fast_list_protection_policies(mixture)
                store.fast_list_replication_streams(mixture)
                mixture_duration = time.time() - mixture_start
                logger.info(f"Finish populating cache for fast mixture={mixture} in {mixture_duration:.2f}s")
            except Exception:
                logger.exception(f"Error populating cache for fast mixture={mixture}")
    
    total_duration = time.time() - start_time
    logger.info(f"Cache population completed in {total_duration:.2f}s")


if __name__ == "__main__":
    main()
