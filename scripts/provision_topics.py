"""Create every Kafka topic the platform uses.

Run once after ``docker compose up -d``::

    python -m scripts.provision_topics

Topics are created with a single partition per symbol group by default, which is
enough for the Phase 2 vertical slice; raise ``--partitions`` when more than one
instance of an agent needs to consume in parallel.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError

from libs.config import settings
from libs.kafka_client import ALL_TOPICS
from libs.logging_config import configure_logging

logger = logging.getLogger("provision_topics")


async def provision(partitions: int, replication: int) -> None:
    """Create any missing topics.

    Args:
        partitions: Partition count for new topics.
        replication: Replication factor for new topics.
    """
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka.bootstrap_servers)
    await admin.start()
    try:
        new_topics = [
            NewTopic(name=topic, num_partitions=partitions, replication_factor=replication)
            for topic in ALL_TOPICS
        ]
        try:
            await admin.create_topics(new_topics)
            logger.info("Created topics", extra={"topics": list(ALL_TOPICS)})
        except TopicAlreadyExistsError:
            logger.info("Topics already exist", extra={"topics": list(ALL_TOPICS)})
    finally:
        await admin.close()


def main() -> None:
    """Parse arguments and provision topics."""
    parser = argparse.ArgumentParser(description="Provision platform Kafka topics.")
    parser.add_argument("--partitions", type=int, default=1, help="Partitions per topic.")
    parser.add_argument("--replication", type=int, default=1, help="Replication factor.")
    args = parser.parse_args()

    configure_logging("provision_topics")
    asyncio.run(provision(args.partitions, args.replication))


if __name__ == "__main__":
    main()
