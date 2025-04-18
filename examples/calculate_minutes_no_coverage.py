from idea.validation.util import update_no_coverage_counters

if __name__ == "__main__":
    counters_zeros = 0
    counters_low = 0
    for minute, fcd in enumerate([0, 0, 1, 2, 0, 0, 0, 1, 0, 2, 0, 0]):
        print(f"minute: {minute} counter zeros: {counters_zeros}, counter low: {counters_low}")
        counters_zeros, counters_low = update_no_coverage_counters(
            fcd, counters_zeros, counters_low
        )
