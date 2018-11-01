


def wash():
    try:
        f_source = open('./data/mscoco/test_target.txt', 'r')
        f_target = open('./data/mscoco/trimed_source_target.txt', 'w')

        for num, line_pair in enumerate(f_source):
            f_target.write(line_pair)
            if num == 161947:
                break
    finally:
        if f_source:
            f_source.close()
        if f_target:
            f_target.close()


if __name__ == '__main__':
    wash()
