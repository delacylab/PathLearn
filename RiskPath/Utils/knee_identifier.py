########################################################################################################################
# Import packages
########################################################################################################################
import numpy as np
########################################################################################################################


def knee(X):
    """
    Using the triangle method to identify the knee of a curve which corresponds to the farther point perpendicular to
    the baseline connecting the two endpoints.
    :param X: A one-dimensional numpy.ndarray.
    :return:
    (a) knee_index: An integer, representing the position (0-indexed) of the knee.
    (b) order: A one-dimensional numpy.ndarray, representing the indices of the descending order.
    """
    try:
        X = np.array(X)
    except TypeError:
        raise TypeError(f'X must be, convertible to, a numpy.ndarray. Now its type is {type(X)}.')
    assert len(X.shape) == 1, f'X must be one-dimensional. Now it is {len(X.shape)}-dimensional.'
    order = np.argsort(X)[::-1]
    X_ordered = X[order]
    values, n_points, all_indices = X_ordered, len(X_ordered), np.arange(len(X_ordered))
    start = np.array([0, values[0]])
    end = np.array([n_points - 1, values[-1]])
    line_vector = end - start
    line_vector_norm = line_vector / np.linalg.norm(line_vector)
    vec_from_start = np.column_stack([all_indices, values]) - start
    projection = np.dot(vec_from_start, line_vector_norm)
    projection_point = np.outer(projection, line_vector_norm) + start
    distances = np.linalg.norm(vec_from_start - projection_point, axis=1)
    knee_index = np.argmax(distances)
    return knee_index, order

########################################################################################################################
