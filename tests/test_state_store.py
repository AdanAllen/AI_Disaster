from cachelib import SimpleCache

from state_store import store_is_ready


def test_overlapping_readiness_probes_do_not_delete_each_others_keys():
    class OverlappingCache(SimpleCache):
        inner_ready = None
        probe_started = False

        def get(self, key):
            if not self.probe_started:
                self.probe_started = True
                self.inner_ready = store_is_ready(self)
            return super().get(key)

    store = OverlappingCache()

    assert store_is_ready(store)
    assert store.inner_ready is True
