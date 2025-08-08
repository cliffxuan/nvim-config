#!/usr/bin/env python3
"""
Script to create flow shirts using ActionDaskShirtNew or ActionDaierShirtNew
Usage: python create_shirt.py [dusk|daier]
"""

import argparse
import logging
import sys
import traceback
import uuid
from datetime import datetime

import dotenv
import yaml

from app.worker_functions.actions.action_daier_shirt_new import ActionDaierShirtNew
from app.worker_functions.actions.action_dusk_shirt_new import ActionDaskShirtNew

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


dotenv.load_dotenv()


def get_dusk_config():
    """Get configuration for DASK shirt creation"""
    return {
        "shirt_name": "drie-test-dusk",
        "oyster": "predusk10",
        "owner": "Flow Shop Management",
        "customId": 9670,
        "metal": "gold",
        "temp": "high",
        "ownerId": "universe/earth/company/3095",
        "job_hard_threshold": "10GB",
        "job_soft_threshold": "8GB",
        "to_delete": True,
        "legacy_paths": [r"\\hedera.net\dfs\avalanche\Data\drie-test-dusk"],
        "ntfs": [
            {"permission": "modify", "trustee": r"X1\lfs--icstest-rw"},
            {"permission": "read", "trustee": r"X1\lfs--icstest-ro"},
        ],
    }


def get_daier_config():
    """Get configuration for Daier shirt creation"""
    return {
        "shirt_name": "drie-test-daier",
        "oyster": "predaier10",
        "owner": "Flow Shop Management",
        "customId": 9670,
        "metal": "gold",
        "temp": "high",
        "ownerId": "universe/earth/company/3095",
        "job_hard_threshold": "10GiB",
        "job_soft_threshold": "8GiB",
        "ntfs": [
            {
                "permission": "modify",
                "trustee": r"X1\lfs--icstest-rw",
                "trustee_type": "group",
            },
            {
                "permission": "read",
                "trustee": r"X1\lfs--icstest-ro",
                "trustee_type": "group",
            },
        ],
    }


def create_dusk_shirt():
    """Create a DASK shirt using the provided configuration"""
    config = get_dusk_config()

    # Generate unique job name using timestamp and UUID
    job_name = (
        f"create-shirt-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:2]}"
    )

    # Set repo path following the same pattern as existing code
    repo_path = f"/tmp/{job_name}/dusk-views"

    try:
        logger.info(f"Creating DASK shirt with job name: {job_name}")
        logger.info(f"Shirt name: {config['shirt_name']}")
        logger.info(f"Oyster: {config['oyster']}")
        logger.info(f"Using repo path: {repo_path}")

        # Initialize ActionDaskShirtNew
        action = ActionDaskShirtNew(repo_path, config, job_name)

        logger.info("Shirt object created successfully")
        logger.info(f"Git path: {action.shirt.git_path}")
        logger.info(f"Shirt path: {action.shirt.shirt_path}")

        # Display the generated shirt configuration
        logger.info("Generated shirt configuration:")
        shirt_dict = action.shirt.todict()

        # Pretty print the configuration
        logger.info(
            "\n" + yaml.dump(shirt_dict, default_flow_style=False, sort_keys=False)
        )

        # Provision the shirt (this will create YAML, handle groups, and create PR)
        logger.info("\nProvisioning shirt...")
        action.provision()

        logger.info("DASK shirt provisioning completed successfully!")

    except Exception as e:
        logger.error(f"Error creating DASK shirt: {e}")
        traceback.print_exc()
        sys.exit(1)


def create_daier_shirt():
    """Create an Daier shirt using the provided configuration"""
    config = get_daier_config()

    # Generate unique job name using timestamp and UUID
    job_name = (
        f"create-shirt-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:2]}"
    )

    # Set repo path following the same pattern as existing code
    repo_path = f"/tmp/{job_name}/daier-shirts"

    try:
        logger.info(f"Creating Daier shirt with job name: {job_name}")
        logger.info(f"Shirt name: {config['shirt_name']}")
        logger.info(f"Oyster: {config['oyster']}")
        logger.info(f"Using repo path: {repo_path}")

        # Initialize ActionDaierShirtNew
        action = ActionDaierShirtNew(repo_path, config, job_name)

        logger.info("Shirt object created successfully")
        logger.info(f"Git path: {action.shirt.git_path}")
        logger.info(f"Shirt path: {action.shirt.shirt_path}")

        # Display the generated shirt configuration
        logger.info("Generated shirt configuration:")
        shirt_dict = action.shirt.todict()

        # Pretty print the configuration
        logger.info(
            "\n" + yaml.dump(shirt_dict, default_flow_style=False, sort_keys=False)
        )

        # Provision the shirt (this will create YAML, handle groups, and create PR)
        logger.info("\nProvisioning shirt...")
        action.provision()

        logger.info("Daier shirt provisioning completed successfully!")

    except Exception as e:
        logger.error(f"Error creating Daier shirt: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Create flow shirts (DASK or Daier)",
        epilog=(
            "Examples:\n  python create_shirt.py dusk\n  python create_shirt.py daier"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "shirt_type",
        choices=["dusk", "daier"],
        help="Type of shirt to create (dusk or daier)",
    )

    args = parser.parse_args()

    logger.info(f"Starting {args.shirt_type.upper()} shirt creation...")

    if args.shirt_type == "dusk":
        create_dusk_shirt()
    elif args.shirt_type == "daier":
        create_daier_shirt()


if __name__ == "__main__":
    main()
