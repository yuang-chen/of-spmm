"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
import oneflow

class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(
            dataset, auto_collation, collate_fn, drop_last
        )
        self.dataset_iter = iter(dataset)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    break
            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(
            dataset, auto_collation, collate_fn, drop_last
        )

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            oneflow._oneflow_internal.profiler.RangePush('4096slice')
            data = [self.dataset[idx] for idx in possibly_batched_index]
            # data = []
            # for idx in possibly_batched_index:
            #     oneflow._oneflow_internal.profiler.RangePush("slice")
            #     data.append(self.dataset[idx])
            #     oneflow._oneflow_internal.profiler.RangePop()
            oneflow._oneflow_internal.profiler.RangePop()
        else:
            oneflow._oneflow_internal.profiler.RangePush('fetch-dataset')
            data = self.dataset[possibly_batched_index]
            oneflow._oneflow_internal.profiler.RangePop
        oneflow._oneflow_internal.profiler.RangePush('collate_fn')
        ret = self.collate_fn(data)
        oneflow._oneflow_internal.profiler.RangePop()
        return ret
