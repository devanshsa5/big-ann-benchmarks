from datetime import timedelta

from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS, download_accelerated
import numpy as np
import psutil
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.diagnostics import ServiceType
from couchbase.exceptions import (CouchbaseException, QueryErrorContext,
                                  QueryIndexNotFoundException
)
from couchbase.management.buckets import BucketManager
from couchbase.management.search import SearchIndex
from couchbase.options import (ClusterOptions, QueryOptions,
                               WaitUntilReadyOptions)

class CouchbaseGSIClient(BaseANN):

    def __init__(self, metric, index_params) -> None:
        self._index_params = index_params
        self.services = [ServiceType.KeyValue, ServiceType.Query]
        if metric == "cosine":
            metric = "COSINE"
        else:
            metric = "L2"
        self._metric = metric

        self._query_bs = -1
        self.description = index_params.get("description", "IVF,SQ8")
        self.dim = int(index_params.get("dim",128))
        host = index_params.get("host")
        self.username = index_params.get("username")
        self.password = index_params.get("password")
        self.ssl_mode = index_params.get("ssl_mode", None)
        self.is_capella = self.ssl_mode == "capella"

        if 'query_bs' in index_params:
            self._query_bs = index_params['query_bs']

        cb_proto = ""
        if self.ssl_mode in ("tls", "capella", "n2n") and "://" not in host:
            cb_proto = "couchbases://"
        elif "://" not in host:
            cb_proto = "couchbase://"

        params = ""
        if cb_proto.startswith("couchbases:") or host.startswith("couchbases:"):
            params = "?ssl=no_verify"

        self.connection_string = f"{cb_proto}{host}{params}"
        self.bucket = index_params.get("bucket", "bucket-1")
        self.index_name = f"{self.bucket}_vector_index2"
        self.index_type = index_params.get("index_type", "CVI")
        self.scope = index_params.get("scope", None)
        self.collection = index_params.get("collection", None)
        
    def _get_cluster(self):
        """Helper for creating a cluster connection."""
        auth = PasswordAuthenticator(self.username, self.password)
        cluster_options = ClusterOptions(auth)
        if self.is_capella:
            cluster_options.apply_profile("wan_development")

        cluster = Cluster(self.connection_string, cluster_options)

        cluster.wait_until_ready(
            timedelta(seconds=30), WaitUntilReadyOptions(service_types=self.services)
        )
        return cluster

    def done(self):
        """
        This is called after results have been processed.
        Use it for cleaning up if necessary.
        """
        pass

    def track(self):
        return "T1"

    def fit(self, dataset):
        print("Fitting")

        create_index_query = self._get_create_index_statement()
        cluster = self._get_cluster()
        try:
            print("Building index")
            cluster.query(
                create_index_query, QueryOptions(timeout=timedelta(seconds=180))
            ).execute()
        except CouchbaseException as e:
            print(e)
            # Possibly a timeout, just continue and wait for the index to be ready

    def load_index(self, dataset):
        return True
    
    def set_query_arguments(self, query_args):
        print('# query_args: {} {}'.format(query_args, type(query_args)))
        self._query_args = query_args
        self.nprobes = int(self._query_args)


    def index_files_to_store(self, dataset):
        """
        Specify a triplet with the local directory path of index files,
        the common prefix name of index component(s) and a list of
        index components that need to be uploaded to (after build)
        or downloaded from (for search) cloud storage.

        For local directory path under docker environment, please use
        a directory under
        data/indices/track(T1 or T2)/algo.__str__()/DATASETS[dataset]().short_name()
        """
        raise NotImplementedError()

    def query(self, X, k):
        print("queries =: {} {}".format(type(X), X.shape))
        print(X)
        I = []
        for query in X:
            rows = [0]
            options = QueryOptions(timeout=timedelta(minutes=5))
            try:
                select_query = f"SELECT meta().id from `{self.bucket}` ORDER BY ANN(emb, {query}, '{self._metric}', {self.nprobes}) LIMIT {k};"
                query_result = self._get_cluster().query(select_query, options).execute()
                rows = [int(row.get("id", 0)) for row in query_result]
            except CouchbaseException as e:
                print(e.message)
                if isinstance(e.error_context, QueryErrorContext):
                    print(e.error_context.response_body)

            I.append(np.array(rows))
        self.res = np.array(I)

    def range_query(self, X, radius):
        """
        Carry out a batch query for range search with
        radius.
        """
        raise NotImplementedError()


    def get_results(self):
        """
        Helper method to convert query results of k-NN search.
        If there are nq queries, returns a (nq, k) array of integers
        representing the indices of the k-NN for each query.
        """
        return self.res

    def get_range_results(self):
        """
        Helper method to convert query results of range search.
        If there are nq queries, returns a triple lims, D, I.
        lims is a (nq) array, such that

            I[lims[q]:lims[q + 1]] in int

        are the indices of the indices of the range results of query q, and

            D[lims[q]:lims[q + 1]] in float

        are the distances.
        """
        return self.res

    def get_additional(self):
        """
        Retrieve additional results.
        Return a dictionary with metrics
        and corresponding measured values.

        The following additional metrics are supported:

        `mean_latency` in microseconds, if this applies to your algorithm.
        Skip if your algorithm batches query processing.

        `latency_999` is the 99.9pc latency in microseconds, if this applies
        to your algorithm. Skip if your algorithm batches query processing.

        `dist_comps` is the total number of points in the base set
        to which a query was compared.

        `mean_ssd_ios` is the average number of SSD I/Os per query for T2 algorithms.
        """
        return {}

    def __str__(self):
        return self.name

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    def index_param(self):
        return {
            "dimension": self.dim,
            "description": self.description,
            "similarity": self._metric,
        }
    
    def _get_create_index_statement(self) -> str:
        index_params = self.index_param()
        prefix = ""
        fields = "emb VECTOR"
        if self.index_type == "BHIVE":
            prefix = "VECTOR"
        bucket_suffix = ""
        if self.scope and self.collection:
            bucket_suffix = f".`{self.scope}`.`{self.collection}`"

        return f"CREATE {prefix} INDEX `{self.index_name}` ON `{self.bucket}`{bucket_suffix}({fields}) USING GSI WITH {index_params}"
