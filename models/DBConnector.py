from typing import List, Dict, Tuple

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.cursor import Cursor


class DBConnector:
    def __init__(self, client: MongoClient):
        self._client = client

    def get_collection(self, db_name: str, collection_name: str, max_node_props: List[str]) -> Collection:
        return DBConnector.Collection(self._client[db_name][collection_name], max_node_props)

    class Collection:
        def __init__(self, collection: Collection, label_counts: List[str]):
            self._sample_collection = collection
            self._label_counts = label_counts

        def get_ids(self) -> List[Tuple[ObjectId, int]]:
            documents = self._get_ids()
            sample_ids = [(document["_id"], self._get_max_label_count(document)) for document in documents]
            return sample_ids

        def get_ids_as_string(self) -> List[Tuple[str, int]]:
            documents = self._get_ids()
            sample_ids = [(str(document["_id"]), self._get_max_label_count(document)) for document in documents]
            return sample_ids

        def _get_max_label_count(self, document: Dict[str, int]) -> int:
            label_counts = [document[label_property] for label_property in self._label_counts]
            return max(label_counts)

        def _get_ids(self) -> Cursor:
            filter_dict = {label_count: {"$ne": None} for (label_count) in self._label_counts}
            return self._sample_collection.find(
                filter=filter_dict,
                projection=["_id"] + self._label_counts)

        def get_samples(self, sample_ids: List[ObjectId]) -> List[Dict]:
            return list(self._sample_collection.find({"_id": {"$in": sample_ids}}))

        def get_first_sample(self) -> Dict:
            return self._sample_collection.find_one()
