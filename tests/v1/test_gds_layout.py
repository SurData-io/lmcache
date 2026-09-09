"""Splitter geometry tests (no GPU)."""

from lmcache.v1.gds_layout import pad_to, split_page, split_pages


def test_split_page_bs16_stripe4096():
    page = 18432
    align = 4096
    # odd page pointer 2048-aligned
    head, body, tail = split_page(2048, page, align)
    assert head == 2048
    assert (head + body + tail) == page
    assert body % align == 0


def test_split_pages_phys_align():
    addrs = [0x1000 + i * 18432 for i in range(4)]
    for ss in (4096, 8192):
        layout = split_pages(addrs, 18432, ss, sector=4096)
        assert layout.geometry_ok()
        assert layout.direct_phys_len % ss == 0
        assert layout.residual_phys_len % ss == 0
        packed = 0
        for s in layout.direct_slices:
            assert s.gpu_addr % layout.align == 0
            assert s.length % layout.align == 0
            packed += s.length
        assert packed == layout.direct_logical_len


def test_pad_to():
    assert pad_to(0, 4096) == 0
    assert pad_to(1, 4096) == 4096
    assert pad_to(4096, 4096) == 4096
