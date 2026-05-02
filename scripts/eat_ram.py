"""Diagnostic: allocate ~N MB of RAM and hold it for K seconds, then
release. Used to simulate a user opening Chrome / Slack on a peer
mid-inference so we can verify the adaptive watchdog reacts to a
free-RAM drop.

    python eat_ram.py 1500 30   # eat 1500 MB for 30 seconds, then release
"""
import gc
import sys
import time


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: eat_ram.py <mb> <hold_seconds>")
        return 2
    mb = int(argv[1])
    hold = int(argv[2])
    # Allocate via bytearray — actually committed pages, not just
    # virtual reservation, so the OS reports the change in
    # available memory immediately.
    print(f"allocating {mb} MB...")
    block = bytearray(mb * 1024 * 1024)
    # Touch every page so Windows actually commits them. Without this,
    # the allocation is pageable and may not show up in
    # FreePhysicalMemory until something pressures it.
    for i in range(0, len(block), 4096):
        block[i] = 1
    print(f"holding for {hold}s")
    time.sleep(hold)
    del block
    gc.collect()
    print("released")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
