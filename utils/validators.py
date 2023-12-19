from sklearn.base import is_classifier

def validate_classifier(estimator) -> bool:
    ''' Check if an estimator is a classifier in sklearn framework

    :param estimator:  sklearn estimator to validate
    :return: boolean for whether the estimator is a classifier
    '''

    return is_classifier(estimator)

