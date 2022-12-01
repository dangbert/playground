import pandas as pd
from numpy import nan
from pandas import DataFrame, Series, testing


def test_1(income):
    print("Testing dtype of answer: ", end='')
    assert type(income) is Series, "Income has to be a Series object"
    print('Success!')
    print("Testing indices: ", end='')
    assert set(income.index) == {"sales", "ads", "subscriptions", "donations"}
    print('Success!')
    print("Testing values: ", end='')
    assert set(income) == {39041, 8702, 13200, 292}
    print('Success!')


def test_2(outcome, profit, total_profit):
    print("Testing dtypes: ", end='')
    assert type(outcome) is Series
    assert type(profit) is Series
    float(total_profit)
    print('Success!')
    print("Testing values: ", end='')
    assert set(profit) == {3989, 292, 14759, 9898}
    assert total_profit == 28938
    print('Success!')


def test_3(skittles):
    print("Testing dtype: ", end='')
    assert type(skittles) is DataFrame
    print('Success!')
    print("Testing labels: ", end='')
    assert list(skittles.index) == ["red", "green", "blue", "purple", "pink"]
    assert list(skittles.columns) == ["amount", "rating"]
    print('Success!')
    print("Testing values: ", end='')
    assert skittles.loc["red", "amount"] == 7
    assert skittles.loc["blue", "rating"] == 2
    print('Success!')


def test_4(skittles_average):
    print("Testing: ", end='')
    float(skittles_average)
    assert skittles_average == 3.3
    print('Success!')


def test_5(df):
    print("Testing: ", end='')
    assert "score" in df
    a005 = DataFrame([[21.0],
                      [16.0],
                      [12.0],
                      [20.0],
                      [21.0]],
                     columns=['score'],
                     index=['red', 'green', 'blue', 'purple', 'pink'])
    testing.assert_series_equal(df["score"], a005["score"])
    print('Success!')


def test_6(frame):
    solution = DataFrame([[0.0, 2.0, 4.0],
                          [7.0, 9.0, 11.0],
                          [28.0, 30.0, 32.0],
                          [35.0, 37.0, 39.0]],
                         columns=['a', 'c', 'e'],
                         index=[10, 20, 50, 60])
    print("Testing: ", end='')
    assert type(frame) == DataFrame
    pd.testing.assert_frame_equal(frame, solution)
    print('Success!')


def test_7(frame):
    solution = DataFrame([[0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 0.0],
                          [7.0, 8.0, 0.0, 10.0, 11.0, 0.0, 13.0],
                          [14.0, 0.0, 16.0, 17.0, 0.0, 19.0, 20.0],
                          [0.0, 22.0, 23.0, 0.0, 25.0, 26.0, 0.0],
                          [28.0, 29.0, 0.0, 31.0, 32.0, 0.0, 34.0],
                          [35.0, 0.0, 37.0, 38.0, 0.0, 40.0, 41.0]],
                         columns=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                         index=[10, 20, 30, 40, 50, 60])

    print("Testing: ", end='')
    assert type(frame) == DataFrame
    pd.testing.assert_frame_equal(frame, solution)
    print('Success!')


def test_8(series_a, series_b, series_c):
    solution = Series([477, 400, 254, 200, 100],
                      index=["a", "b", "c", "d", "e"])
    solution = solution.astype("float64")

    print("Testing: ", end='')
    assert all(type(s) is Series for s in [series_a, series_b, series_c])
    pd.testing.assert_index_equal(series_a.index, series_c.index)
    series_c = series_c.astype('float64')
    pd.testing.assert_series_equal(series_c, solution)
    print('Success!')


def test_9(words):
    print("Testing: ", end='')
    assert type(words) is Series
    assert all(isinstance(word, str) for word in words)
    pd.testing.assert_series_equal(
        Series(["foo", "bar", "baz", "qux", "quuux"]), words)
    print('Success!')


def test_10(frame):
    solution = DataFrame(
        [[1.75378, 2.409629, 1.842674, 0.754906, -0.115614,
          0.877219, 1.599362],
         [-1.41176, 1.103801, 1.216514, 0.548866, 2.255482,
          -0.176342, 0.965265],
         [-0.741689, 0.216645, -0.278025, 0.777175, 0.869239,
          -0.943004, -0.140957],
         [0.58498, 0.62389, -0.425614, 0.530479, -1.818631,
          -1.593188, 1.591233],
         [-1.58593, 1.1796, -0.702286, 2.367875, 0.592748,
          1.386158, 0.535978],
         [0.691074, -1.272521, -0.968045, -2.066171,
          -0.670358, 1.399483, -1.148168]],
        columns=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
        index=[1, 2, 3, 5, 4, 0])

    print("Testing: ", end='')
    assert type(frame) is DataFrame
    pd.testing.assert_frame_equal(frame, solution)
    print('Success!')


def test_11(speeds):
    solution = Series(
        [49.0, 51.0, 51.0, 51.0, 50.0, 48.0, 47.0, 50.0, 51.0, 47.0, 46.0,
         46.0, 46.0, 48.0, 48.0, 48.0, 48.0, 49.0, 49.0, 49.0, 49.0, 50.0,
         50.0, 50.0, 51.0, 52.0, 51.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
         50.0, 49.0, 48.0, 49.0, 49.0, 50.0, 50.0, 49.0],
        index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
               18, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37,
               38, 39, 40, 41, 42, 43, 44]).astype("float64")
    print("Testing: ", end='')
    assert type(speeds) is Series
    pd.testing.assert_series_equal(speeds, solution)
    print('Success!')
    
def test_12(average_course_grade, student_count):
    average_course_grade_solution = Series([7.375, 6.5], index=['Programming 1', 'Programming 2']).astype("float64")
    student_count_solution = Series([4, 3], index=['Programming 1', 'Programming 2']).astype("int64")
    
    print("Testing: ", end='')
    assert type(average_course_grade) is Series
    pd.testing.assert_series_equal(average_course_grade, average_course_grade_solution, check_names=False)
    
    assert type(student_count) is Series
    pd.testing.assert_series_equal(student_count, student_count_solution, check_names=False)
    print('Success!')


def test_13(top_students):
    solution = DataFrame([['Ursula', 'Programming 1', 9.5],
                      ['Ursula', 'Programming 2', 9.0],
                      ['Marge', 'Programming 1', 8.0],
                      ['Pascal', 'Programming 2', 7.0]],                      
                      columns=['student_name', 'course_name', 'grade'],                      
                      index=[3, 6, 5, 0])
    
    print("Testing: ", end='')
    pd.testing.assert_frame_equal(top_students, solution)
    print('Success!')

def test_14(pivot_grades):
    solution = DataFrame([[8.0, nan],
                      [5.5, 3.5],
                      [nan, 7.0],
                      [6.5, nan],
                      [9.5, 9.0]],                      
                      columns=['Programming 1', 'Programming 2'],                      
                      index=['Marge', 'Morty', 'Pascal', 'Slartibartfast', 'Ursula'])
    print("Testing: ", end='')
    pd.testing.assert_frame_equal(pivot_grades, solution, check_names=False)
    print('Success!')    
    

def test15(exploded_actors):
    pass
    
    
def test_16():
    print("Testing: ", end='')
    
    student_solution = pd.read_csv('data/ingredients.csv').sort_values(['ingredients', 'recipe_name']).reset_index(drop=True)
    solution = pd.read_csv('data/ingredients_solution.csv').sort_values(['ingredients', 'recipe_name']).reset_index(drop=True)
    
    
    pd.testing.assert_frame_equal(student_solution, solution)
    print('Success!')
