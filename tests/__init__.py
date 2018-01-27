
def check_metadata(true_metadata, test_metadata):
    # Perform comparising as true_metadata belongs to test_metadata,
    # test_metadata can contain other keys
    for k in true_metadata:
        if k not in test_metadata:
            return False
    return True
