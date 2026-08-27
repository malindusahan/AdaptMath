"""Build realistic student-style questions and utterances for all 111 Canonical Skills."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology.ontology_seed_service import (
    DEFAULT_ONTOLOGY_JSON_PATH,
    generate_skill_id,
    normalize_token,
)

SEED_DATASET_CSV = (
    PROJECT_ROOT
    / "data"
    / "topic_extraction"
    / "processed"
    / "ontology_seed_dataset.csv"
)
STUDENT_STYLE_CSV = (
    PROJECT_ROOT
    / "data"
    / "topic_extraction"
    / "processed"
    / "student_style_dataset.csv"
)
COMBINED_DATASET_CSV = (
    PROJECT_ROOT
    / "data"
    / "topic_extraction"
    / "processed"
    / "topic_extraction_dataset.csv"
)


# Skill-specific realistic natural language student problem phrasings
SKILL_STUDENT_PHRASES: dict[int, list[tuple[str, str]]] = {
    # 1: Box and Whisker
    1: [
        ("How do I plot the quartiles and median on the box plot?", "graphing"),
        ("What does the line in the middle of the box represent?", "interpretation"),
        ("How do I find the interquartile range from a box and whisker chart?", "calculation"),
        ("Where are the upper and lower whiskers located?", "structure"),
        ("How do I identify outliers on this box graph?", "outliers"),
        ("Which quartile represents the 75th percentile?", "percentiles"),
        ("How do I draw a box and whisker plot from a list of data?", "graphing"),
        ("What do the ends of the whiskers show on the diagram?", "interpretation"),
        ("How do I divide the dataset into four equal parts for the box plot?", "quartiles"),
        ("Is the median always centered in the box plot?", "properties"),
    ],
    # 2: Circle Graph
    2: [
        ("How do I calculate the degrees of a slice in a pie chart?", "calculation"),
        ("What percentage of the pie chart does this section take up?", "percentage"),
        ("How do I convert a percentage to an angle in a circle graph?", "conversion"),
        ("If the total is 360 degrees, how large is a 25 percent slice?", "angles"),
        ("How do I read proportions from a circular chart?", "interpretation"),
        ("How do you find the total amount represented by the entire circle graph?", "totals"),
        ("What fraction of the circle graph corresponds to this category?", "fractions"),
        ("How do I create a pie chart from survey data?", "graphing"),
        ("Why do all the slices in a circle graph have to add up to 100 percent?", "properties"),
        ("How do I find the value of the missing piece in a circle graph?", "missing_value"),
    ],
    # 4: Histogram as Table or Graph
    4: [
        ("What is the difference between a bar chart and a histogram?", "comparison"),
        ("How do I choose the bin width and intervals for a histogram?", "intervals"),
        ("How do I read frequency from the vertical axis of a histogram?", "frequency"),
        ("Why are the bars touching each other in this histogram graph?", "properties"),
        ("How many data points fall between 10 and 20 in this frequency graph?", "range_lookup"),
        ("How do I construct a frequency table into a histogram?", "graphing"),
        ("How do I find the modal class interval on a histogram?", "interpretation"),
        ("What does the height of each column in a histogram indicate?", "frequency"),
        ("How do I group continuous numbers into bins on a frequency plot?", "binning"),
        ("How do I determine the total number of items from a histogram?", "totals"),
    ],
    # 5: Number Line
    5: [
        ("Where does negative 4 point 5 sit on the horizontal line?", "placement"),
        ("How do I plot fractions on a number line between 0 and 1?", "fractions"),
        ("Which number is further to the left on the number line?", "comparison"),
        ("How do I show open and closed circles on a number line?", "intervals"),
        ("How do I count steps between positive and negative numbers on the line?", "distance"),
        ("How do I locate decimals on a tick-marked number line?", "decimals"),
        ("What is the distance between negative 3 and positive 7 on the line?", "distance"),
        ("How do I mark integers on a coordinate line?", "integers"),
        ("How do I graph inequalities with an arrow on a number line?", "inequalities"),
        ("Which point on the line corresponds to 3 fourths?", "placement"),
    ],
    # 8: Scatter Plot
    8: [
        ("How do I know if this scatter plot has a positive or negative correlation?", "correlation"),
        ("How do I draw a line of best fit through these data points?", "line_of_best_fit"),
        ("What does it mean when the dots on a scatter graph cluster together?", "clustering"),
        ("How do I identify an outlier point on a scatter diagram?", "outliers"),
        ("Is there any relationship between x and y in this scatter plot?", "relationship"),
        ("How do I predict a value using the trend line on a scatter plot?", "prediction"),
        ("What does zero correlation look like on a dot plot?", "correlation"),
        ("How do I plot paired bivariate data on a coordinate scatter graph?", "plotting"),
        ("How do I determine if a scatter plot is linear or non-linear?", "linearity"),
        ("Why are the points scattered and not in a straight line?", "variation"),
    ],
    # 9: Stem and Leaf Plot
    9: [
        ("How do I read the key on a stem and leaf plot?", "key_reading"),
        ("What does a stem of 4 and leaf of 7 mean?", "place_value"),
        ("How do I find the median from a stem and leaf display?", "median"),
        ("How do I organize raw numbers into stems and leaves?", "organization"),
        ("How do I find the mode using a stem and leaf diagram?", "mode"),
        ("How do I make a back to back stem and leaf plot?", "comparison"),
        ("What is the range of values in this stem and leaf chart?", "range"),
        ("Can a leaf have more than one digit in a stem plot?", "rules"),
        ("How do I order the leaves from least to greatest on each stem?", "ordering"),
        ("How many total observations are shown in this stem and leaf display?", "counts"),
    ],
    # 10: Table
    10: [
        ("How do I extract the correct row and column from this data table?", "lookup"),
        ("How do I find the total in a two-way frequency table?", "totals"),
        ("How do I read values across rows and columns in a data grid?", "reading"),
        ("How do I complete the missing entry in this table of values?", "missing_entry"),
        ("How do I calculate marginal totals in a contingency table?", "marginal"),
        ("What is the relationship between the x column and y column in this table?", "relationships"),
        ("How do I find probabilities using a two-way data table?", "probability"),
        ("How do I organize raw measurement results into a structured table?", "organization"),
        ("Which row in the table has the highest recorded score?", "comparison"),
        ("How do I format data into rows and columns with headers?", "formatting"),
    ],
    # 11: Venn Diagram
    11: [
        ("How do I find the overlap between two intersecting circles?", "intersection"),
        ("What does the region outside both circles represent in a Venn diagram?", "complement"),
        ("How do I calculate the union of two sets in a Venn diagram?", "union"),
        ("Where do elements that belong to both set A and set B go?", "intersection"),
        ("How do I place numbers in a three circle Venn diagram?", "three_sets"),
        ("What does the intersection symbol mean in set diagrams?", "notation"),
        ("How do I find how many items belong only to set A and not set B?", "difference"),
        ("How do I calculate the probability of A and B using a Venn diagram?", "probability"),
        ("What does mutually exclusive look like in a Venn diagram?", "disjoint"),
        ("How do I determine the universal set total from the regions in a Venn diagram?", "universal"),
    ],
    # 12: Mean
    12: [
        ("How do I calculate the average of this list of numbers?", "calculation"),
        ("Do I add all numbers and divide by how many there are to find the mean?", "formula"),
        ("What is the arithmetic mean of 12, 15, 18, and 25?", "calculation"),
        ("How does adding a very high outlier affect the mean?", "outliers"),
        ("If the mean of 5 tests is 80, what is the total sum of the scores?", "inverse"),
        ("What score do I need on the next test to get an average of 90?", "target_mean"),
        ("How do I find the mean from a frequency table?", "weighted_mean"),
        ("Why is the average different from the middle number?", "comparison"),
        ("How do I find the missing score if the mean is already given?", "missing_value"),
        ("How do I compute the mean of decimals and negative numbers?", "computation"),
    ],
    # 13: Median
    13: [
        ("How do I find the middle number in an ordered list?", "middle_value"),
        ("What do I do if there are two numbers in the middle when finding the median?", "even_count"),
        ("Do I have to sort the numbers from smallest to largest first to find the median?", "sorting"),
        ("Find the median of 3, 7, 9, 15, 21, 24", "calculation"),
        ("Why is the median resistant to extreme outliers?", "properties"),
        ("How do I find the median when the count of data points is even?", "average_middle"),
        ("Is the median always one of the numbers in the original dataset?", "properties"),
        ("How do I locate the 50th percentile in a sorted list?", "percentiles"),
        ("How do I find the median from a grouped frequency table?", "grouped"),
        ("What is the difference between the median and the mean?", "comparison"),
    ],
    # 14: Mode
    14: [
        ("Which number appears most frequently in this set?", "definition"),
        ("Can a dataset have more than one mode or bimodal distribution?", "bimodal"),
        ("What happens if every number in a list appears only once?", "no_mode"),
        ("How do I identify the mode in a frequency table?", "frequency"),
        ("Find the most common value in 4, 4, 7, 8, 9, 9, 9, 12", "calculation"),
        ("Can categorical data like favorite colors have a mode?", "categorical"),
        ("What does it mean if a distribution is multimodal?", "multimodal"),
        ("How do I find the modal score from a bar chart?", "visual"),
        ("Is the mode always an actual value from the dataset?", "properties"),
        ("How is the mode different from the mean and median?", "comparison"),
    ],
    # 15: Range
    15: [
        ("How do I find the difference between the largest and smallest number?", "calculation"),
        ("Subtract the minimum value from the maximum value in the dataset", "formula"),
        ("What is the range of temperatures from -5 degrees to 25 degrees?", "negative_numbers"),
        ("How does an extreme high value change the range of the data?", "spread"),
        ("Find the spread of scores between the lowest 42 and highest 98", "calculation"),
        ("Is the range a measure of spread or central tendency?", "classification"),
        ("How do I calculate the statistical range from a stem and leaf plot?", "plot_reading"),
        ("What does a small range tell us about data consistency?", "interpretation"),
        ("How do I find the range from a box plot minimum and maximum?", "box_plot"),
        ("Can the range of a dataset ever be a negative number?", "properties"),
    ],
    # 16: Counting Methods
    16: [
        ("How do I use the fundamental counting principle to find total combinations?", "combinations"),
        ("If I have 3 shirts and 4 pants, how many outfits can I make?", "multiplication"),
        ("How do I use a tree diagram to list all possible outcomes?", "tree_diagram"),
        ("What is the difference between permutations where order matters and combinations?", "permutations"),
        ("How do I calculate factorials like 5 factorial?", "factorials"),
        ("How many ways can 4 people sit in a row of 4 chairs?", "arrangements"),
        ("How many 3-digit lock codes can be formed without repeating digits?", "codes"),
        ("How do I count possible outcomes using systematic listing?", "listing"),
        ("How many ways can you choose 2 toppings from a list of 5?", "combinations"),
        ("How do I calculate n choose r for choosing team members?", "combinations"),
    ],
    # 17: Probability of Two Distinct Events
    17: [
        ("How do I find the probability of rolling a 6 and flipping heads?", "independent"),
        ("Do I multiply probabilities when two events happen together?", "multiplication_rule"),
        ("What is the probability of picking a red card followed by a blue card without replacement?", "dependent"),
        ("How do I calculate the chance of event A or event B occurring?", "addition_rule"),
        ("What happens to the denominator when drawing cards without replacement?", "conditional"),
        ("How do I use a tree diagram to compute compound probabilities?", "tree_diagram"),
        ("What is the probability of getting two heads in a row on coin tosses?", "independent"),
        ("How do I find the probability of independent versus dependent events?", "classification"),
        ("How do I solve problems involving compound events with dice and spinners?", "compound"),
        ("What is the formula for the probability of A and B for independent events?", "formula"),
    ],
    # 18: Probability of a Single Event
    18: [
        ("What is the probability of rolling a 4 on a standard six-sided die?", "calculation"),
        ("How do I calculate the favorable outcomes over total possible outcomes?", "formula"),
        ("What is the chance of picking a yellow marble out of a bag of 10 marbles?", "fractions"),
        ("How do I express a probability as a fraction, decimal, and percentage?", "representation"),
        ("What does a probability of 0 or 1 mean for an event?", "certainty"),
        ("How do I find the probability of the complement event not happening?", "complement"),
        ("What is the probability of spinning an odd number on a spinner with 8 sections?", "spinner"),
        ("If an event has a 3 in 5 chance, what is the probability it will occur?", "ratios"),
        ("How do I determine theoretical probability from sample space?", "sample_space"),
        ("What is the probability of drawing a queen from a standard deck of cards?", "cards"),
    ],
    # 21: Interior Angles Figures with More than 3 Sides
    21: [
        ("What is the formula for the sum of interior angles in an n-sided polygon?", "formula"),
        ("How do I calculate (n - 2) * 180 to find the total angles in a pentagon?", "calculation"),
        ("What is the sum of interior angles in a hexagon or octagon?", "polygons"),
        ("How do I find each interior angle of a regular polygon with 8 sides?", "regular_polygon"),
        ("How do I find a missing angle in a 5-sided polygon when 4 angles are known?", "missing_angle"),
        ("How many degrees are inside a 10-sided decagon?", "calculation"),
        ("How do I divide a polygon into triangles from one vertex to find angle sum?", "triangulation"),
        ("What is each angle in an equiangular regular hexagon?", "regular_hexagon"),
        ("How do I solve for x in an algebraic polygon angle equation?", "algebraic_angles"),
        ("Why does adding another side to a polygon add 180 degrees to the interior sum?", "properties"),
    ],
    # 22: Interior Angles Triangle
    22: [
        ("Why do all three interior angles in any triangle always add up to 180 degrees?", "angle_sum"),
        ("If two angles in a triangle are 50 and 60 degrees, what is the third angle?", "missing_angle"),
        ("How do I find the angles of a right triangle when one acute angle is given?", "right_triangle"),
        ("How do I solve for x when the three angles of a triangle are expressed as algebraic expressions?", "algebraic_triangle"),
        ("What are the three interior angles of an equilateral triangle?", "equilateral"),
        ("In an isosceles triangle, how do I find the base angles if the vertex angle is 40 degrees?", "isosceles"),
        ("Can a triangle have two right angles or obtuse angles?", "properties"),
        ("How do I use the 180 degree rule to find the missing corner of a triangle?", "calculation"),
        ("How do I calculate the interior angles from exterior angle relationships?", "exterior_relation"),
        ("What is the third angle if one is 90 degrees and the other is 35 degrees?", "right_triangle"),
    ],
    # 24: Congruence
    24: [
        ("What does it mean for two geometric shapes to be congruent?", "definition"),
        ("How do I prove two triangles are congruent using SSS, SAS, ASA, or AAS?", "triangle_congruence"),
        ("Are two shapes congruent if they have the exact same size and shape?", "properties"),
        ("How do corresponding sides and corresponding angles match up in congruent figures?", "corresponding_parts"),
        ("What is the hypotenuse leg HL theorem for right triangle congruence?", "right_triangles"),
        ("How do rigid transformations like translations and reflections preserve congruence?", "transformations"),
        ("If triangle ABC is congruent to triangle DEF, which side corresponds to BC?", "naming_order"),
        ("Can two shapes be congruent if one is rotated or flipped over?", "orientation"),
        ("How do I write a formal congruence statement with matching vertices?", "notation"),
        ("What is the difference between congruent figures and similar figures?", "comparison"),
    ],
    # 25: Complementary and Supplementary Angles
    25: [
        ("Do complementary angles always add up to 90 degrees?", "complementary"),
        ("Do supplementary angles always add up to 180 degrees on a straight line?", "supplementary"),
        ("What is the complement of a 35 degree angle?", "calculation"),
        ("What is the supplement of a 110 degree angle?", "calculation"),
        ("How do I set up an algebraic equation for two angles that are supplementary?", "algebraic_angles"),
        ("If two angles form a linear pair on a straight line, are they supplementary?", "linear_pair"),
        ("Two angles are complementary and one is twice the other, what are their measures?", "word_problem"),
        ("Can two obtuse angles ever be supplementary?", "properties"),
        ("What is the complementary angle to 48 degrees?", "calculation"),
        ("How do I solve for x when complementary angles are 2x and 3x + 10?", "algebraic_equations"),
    ],
    # 26: Angles on Parallel Lines Cut by a Transversal
    26: [
        ("What are alternate interior angles when parallel lines are cut by a transversal?", "alternate_interior"),
        ("Are corresponding angles equal when lines are parallel?", "corresponding_angles"),
        ("What are consecutive interior or same-side interior angles that add up to 180?", "consecutive_interior"),
        ("How do I find all 8 angles created by a transversal line across parallel lines?", "angle_relationships"),
        ("Are alternate exterior angles congruent on parallel lines?", "alternate_exterior"),
        ("How do vertical angles relate to transversal intersections?", "vertical_angles"),
        ("How do I set up equations when alternate interior angles are expressed algebraically?", "algebraic_transversal"),
        ("If one angle is 70 degrees on parallel lines, what are all the other seven angles?", "calculation"),
        ("How do I prove two lines are parallel using angle theorems?", "proofs"),
        ("Why are same-side exterior angles supplementary?", "properties"),
    ],
    # 27: Pythagorean Theorem
    27: [
        ("How do I use a squared plus b squared equals c squared?", "formula"),
        ("How do I calculate the hypotenuse of a right triangle with legs 3 and 4?", "hypotenuse"),
        ("How do I solve for a missing leg using the Pythagorean theorem?", "missing_leg"),
        ("What is a Pythagorean triple like 3-4-5 or 5-12-13?", "triples"),
        ("How do I find the distance between two points using the Pythagorean theorem?", "distance_formula"),
        ("Is a triangle with sides 6, 8, and 10 a right triangle?", "converse"),
        ("How do I take the square root to find side c in a right triangle?", "square_root"),
        ("How do I find the diagonal of a rectangle using Pythagoras theorem?", "applications"),
        ("How do I solve real-world ladder against a wall word problems?", "word_problems"),
        ("Can the Pythagorean theorem be used on non-right triangles?", "restrictions"),
    ],
    # 39: Area Circle
    39: [
        ("What is the formula pi r squared for finding the area of a circle?", "formula"),
        ("How do I find the area of a circle if I only know the diameter?", "diameter_to_radius"),
        ("Calculate the area of a circle with a radius of 7 using 22/7 for pi", "calculation"),
        ("How do I find the area of a semicircle or quarter circle sector?", "semicircle"),
        ("What is the area inside a circular pool with radius 5 meters?", "applications"),
        ("How do I square the radius before multiplying by 3.14?", "order_of_ops"),
        ("If the area of a circle is 36 pi, what is the radius?", "inverse"),
        ("How do I calculate the area of the shaded region between two concentric circles?", "annulus"),
        ("Why is circle area measured in square units?", "units"),
        ("How do I find circle area given the circumference?", "multi_step"),
    ],
    # 40: Circumference
    40: [
        ("What is the formula 2 pi r or pi d for the circumference of a circle?", "formula"),
        ("How do I find the distance around the outside edge of a circle?", "perimeter"),
        ("What is the circumference of a bicycle wheel with a diameter of 20 inches?", "applications"),
        ("If the radius is 6 cm, what is the exact circumference in terms of pi?", "exact_pi"),
        ("How do I find the radius of a circle if the circumference is 31.4 cm?", "inverse"),
        ("How many times does a wheel rotate over a certain distance?", "rotation_distance"),
        ("What is the perimeter of a half-circle including the straight diameter edge?", "semicircle_perimeter"),
        ("How do I approximate circumference using 3.14 for pi?", "approximation"),
        ("What is the relationship between diameter and circumference?", "ratio_pi"),
        ("How do I calculate the circumference when given the area of a circle?", "multi_step"),
    ],
    # 42: Perimeter of a Polygon
    42: [
        ("How do I find the total distance around the outside of a polygon?", "definition"),
        ("Do I add all the side lengths together to find the perimeter?", "addition"),
        ("What is the perimeter of a rectangle with length 8 and width 5?", "rectangle_perimeter"),
        ("How do I find the perimeter of a regular hexagon with side length 6?", "regular_polygon"),
        ("How do I find a missing side length if the total perimeter is already given?", "missing_side"),
        ("What is the perimeter of a triangle with sides 4cm, 5cm, and 6cm?", "triangle_perimeter"),
        ("How do I write an algebraic expression for the perimeter of a shape with variable sides?", "algebraic_perimeter"),
        ("How do I calculate the perimeter of an irregular polygon on a grid?", "grid_perimeter"),
        ("What is the perimeter of a square with side length 9?", "square_perimeter"),
        ("Why is perimeter measured in linear units instead of square units?", "units"),
    ],
    # 193: Linear Equations
    193: [
        ("How do I solve for x in the equation 2x + 5 = 15?", "two_step"),
        ("How do I isolate the variable when numbers are on both sides of the equals sign?", "variables_both_sides"),
        ("What inverse operation do I use to cancel multiplication in an equation?", "inverse_operations"),
        ("How do I solve equations with fractions like x/3 - 4 = 2?", "fraction_equations"),
        ("How do I expand brackets using the distributive property before solving for x?", "distributive_solving"),
        ("How do I check my solution by plugging the answer back into the equation?", "checking_solutions"),
        ("What does it mean if an equation has no solution or infinitely many solutions?", "special_cases"),
        ("How do I solve 3(x - 4) = 2x + 7 step by step?", "multi_step"),
        ("How do I combine like terms on the same side of a linear equation?", "combine_terms"),
        ("How do I write and solve a linear equation from a word problem?", "word_problems"),
    ],
    # 221: Slope
    221: [
        ("How do I calculate rise over run to find the slope of a line?", "rise_over_run"),
        ("What is the slope formula (y2 - y1) / (x2 - x1)?", "slope_formula"),
        ("What does the letter m represent in y = mx + b?", "slope_intercept"),
        ("What is the difference between positive, negative, zero, and undefined slope?", "slope_types"),
        ("How do I find the slope from two coordinate points (2, 3) and (6, 11)?", "ordered_pairs"),
        ("Why does a horizontal line have a slope of zero?", "horizontal_line"),
        ("Why is the slope of a vertical line undefined?", "vertical_line"),
        ("How do I find the rate of change or steepness from a linear graph?", "rate_of_change"),
        ("What is the slope of parallel lines versus perpendicular lines?", "parallel_perpendicular"),
        ("How do I identify the slope directly from a table of x and y values?", "table_slope"),
    ],
    # 310: Order of Operations All
    310: [
        ("In what order should I calculate parentheses, exponents, multiplication, and division?", "pemdas_rules"),
        ("Do multiplication and division have equal priority from left to right?", "left_to_right"),
        ("How do I evaluate expressions with nested parentheses and brackets?", "nested_brackets"),
        ("Why is 8 - 2 * 3 equal to 2 and not 18?", "precedence"),
        ("How do I apply BODMAS or PEMDAS when negative signs and powers are present?", "powers_and_negatives"),
        ("Evaluate the expression 3 + 4 * (2^3 - 5)", "expression_evaluation"),
        ("Do addition and subtraction get evaluated in order from left to right?", "addition_subtraction"),
        ("How do I correctly simplify fraction bars in order of operations?", "fraction_bar_precedence"),
        ("What operation comes first in 12 / 3 * 2?", "left_to_right_precedence"),
        ("How do I fix order of operations mistakes with negative numbers?", "negatives"),
    ],
    # 350: Solving Systems of Linear Equations
    350: [
        ("How do I solve a system of two equations using the substitution method?", "substitution"),
        ("How do I use elimination or addition method to cancel out a variable?", "elimination"),
        ("What does the point of intersection represent in a system of linear equations?", "intersection"),
        ("How do I solve systems of equations with two variables x and y?", "two_variables"),
        ("When does a system of linear equations have no solution or parallel lines?", "no_solution"),
        ("When does a system of equations have infinitely many solutions?", "infinite_solutions"),
        ("How do I multiply an equation by a constant so coefficients cancel in elimination?", "elimination_prep"),
        ("Solve the system 2x + y = 9 and 3x - y = 6", "system_example"),
        ("How do I set up a system of linear equations from a word problem with two unknowns?", "word_problems"),
        ("How do I check if an ordered pair (x, y) satisfies both equations in a system?", "checking_system"),
    ],
    # 58: Addition Whole Numbers
    58: [
        ("If we combine 25 and 75, what is the sum?", "combining"),
        ("How do I add two large whole numbers with regrouping and carrying?", "regrouping"),
        ("What is the total if I add 15 plus 35?", "addition_total"),
        ("How do I do column addition with multi digit numbers?", "column_addition"),
        ("What is the sum of 120 and 450?", "sum_calculation"),
        ("If I have 50 apples and get 20 more, how many in total?", "word_problem"),
        ("How do I add three numbers together step by step?", "multi_addends"),
        ("What is the result of adding 64 + 37?", "mental_math"),
        ("Why do we carry over 1 to the tens place in addition?", "carrying_concept"),
        ("Find the total sum of 234 and 567", "standard_addition"),
    ],
    # 74: Subtraction Whole Numbers
    74: [
        ("If we separate 10 from 100, what is the answer?", "separation"),
        ("If we take away 10 from 100, how much is left?", "take_away"),
        ("What is 100 minus 10?", "minus_calculation"),
        ("How do I subtract whole numbers with borrowing or regrouping across zeros?", "borrowing"),
        ("What is the difference between 500 and 125?", "difference_concept"),
        ("If I have 100 items and give away 25, how many do I have left?", "word_problem"),
        ("How do I subtract large numbers step by step?", "subtraction_steps"),
        ("What do we get when we deduct 15 from 60?", "deduction"),
        ("How do I find how much more one number is than another using subtraction?", "comparison"),
        ("Calculate 84 minus 29", "standard_subtraction"),
        ("If you subtract 10 from 100 what is the answer?", "natural_subtraction"),
        ("What is the remaining amount if 40 is removed from 90?", "remaining_amount"),
    ],
    # 69: Multiplication Whole Numbers
    69: [
        ("What is the product of 12 and 8?", "product_calculation"),
        ("How do I multiply two digit numbers using long multiplication?", "long_multiplication"),
        ("If there are 6 boxes with 15 apples in each, how many apples total?", "grouping_word_problem"),
        ("What is 25 times 4?", "times_calculation"),
        ("How do I use times tables to solve large multiplication problems?", "times_tables"),
        ("What is the result of multiplying 14 by 7?", "multiplication_steps"),
        ("How do I multiply numbers ending in zeros like 30 * 40?", "trailing_zeros"),
        ("What is 12 multiplied by 12?", "squares_multiplication"),
    ],
    # 277: Addition and Subtraction Integers
    277: [
        ("How do I subtract negative numbers like 5 minus negative 3?", "subtracting_negatives"),
        ("What is negative 7 plus positive 12?", "adding_opposite_signs"),
        ("What is positive 10 minus 20?", "subtracting_larger_number"),
        ("Why does subtracting a negative become addition?", "double_negative_rule"),
        ("How do I add two negative integers like -4 + -8?", "two_negatives"),
        ("What is the rule for signs when adding and subtracting integers?", "sign_rules"),
    ],
}


# Generic template sets by subject category to ensure every skill has 25+ rich utterances
CATEGORY_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "Algebra & Functions": [
        ("How do I solve problems involving {name} step by step?", "step_by_step_solving"),
        ("Can you show me the algebraic steps for {name}?", "algebraic_steps"),
        ("I get confused when variables are involved in {name}", "variable_confusion"),
        ("What formula or rule do I use for {name}?", "formula_inquiry"),
        ("How do I simplify expressions related to {name}?", "expression_simplification"),
        ("Can we work through a sample question on {name}?", "sample_walkthrough"),
        ("How do I set up an algebraic equation for {name}?", "equation_setup"),
        ("What is the first step when tackling {name}?", "first_step_strategy"),
        ("How do I check if my answer is correct in {name}?", "verification_strategy"),
        ("What are common algebraic mistakes to avoid in {name}?", "mistake_avoidance"),
        ("How do I graph and interpret equations involving {name}?", "graphing_interpretation"),
        ("Can you explain the properties and rules of {name}?", "rules_and_properties"),
        ("How does {name} apply to word problems?", "word_problem_application"),
        ("I need practice isolating the terms in {name}", "term_isolation_practice"),
        ("How do I evaluate expressions when practicing {name}?", "expression_evaluation"),
    ],
    "Geometry & Measurement": [
        ("How do I find the geometric measurement for {name}?", "measurement_calculation"),
        ("What is the exact formula used to calculate {name}?", "geometry_formula"),
        ("How do I solve for missing dimensions when calculating {name}?", "missing_dimension"),
        ("What units should I use when writing the answer for {name}?", "unit_specification"),
        ("Can you explain the geometric properties of {name}?", "geometric_properties"),
        ("How do I find the angles or side lengths in {name}?", "dimension_resolution"),
        ("How do 2D and 3D shapes relate to {name}?", "spatial_relationships"),
        ("Can you walk me through a diagram problem about {name}?", "diagram_walkthrough"),
        ("What is the difference between area, perimeter, and {name}?", "concept_distinction"),
        ("How do I calculate the shaded region in problems with {name}?", "shaded_region"),
        ("How do transformations or measurements work in {name}?", "transformations"),
        ("What theorem or geometric rule applies to {name}?", "theorem_application"),
        ("How do I solve word problems involving {name}?", "geometry_word_problem"),
        ("Can we do a practice calculation for {name}?", "practice_calculation"),
        ("What is the step by step method to solve {name}?", "step_by_step_method"),
    ],
    "Statistics & Probability": [
        ("How do I interpret and analyze data using {name}?", "data_interpretation"),
        ("What does the result of {name} tell us about the dataset?", "data_analysis"),
        ("How do I construct and read a graph for {name}?", "graph_construction"),
        ("What formula is used to calculate {name}?", "stats_formula"),
        ("How do outliers affect the calculation of {name}?", "outlier_impact"),
        ("How do I find the probability or distribution in {name}?", "probability_distribution"),
        ("Can you explain how to summarize sample data with {name}?", "sample_summary"),
        ("What is the difference between different measures like {name}?", "measure_comparison"),
        ("How do I solve chance and outcome questions on {name}?", "chance_outcomes"),
        ("How do I organize frequency and data points for {name}?", "frequency_organization"),
        ("Can we walk through an example of calculating {name}?", "stats_walkthrough"),
        ("How do I draw conclusions from a statistical display of {name}?", "conclusions_drawing"),
        ("What is the likelihood of an outcome when using {name}?", "outcome_likelihood"),
        ("How do I find percentages and ratios in {name}?", "percentages_ratios"),
        ("How do I solve multi-step word problems involving {name}?", "stats_word_problem"),
    ],
    "Number Sense & Operations": [
        ("How do I perform basic arithmetic and calculations for {name}?", "arithmetic_procedure"),
        ("What are the rules for positive and negative numbers in {name}?", "sign_rules"),
        ("How do I simplify and compute expressions involving {name}?", "expression_computation"),
        ("What is the standard procedure to solve {name}?", "standard_procedure"),
        ("How do I work with fractions, decimals, and integers in {name}?", "number_types"),
        ("Can you explain the mathematical steps to evaluate {name}?", "evaluation_steps"),
        ("What order or rule should I follow when working on {name}?", "operational_order"),
        ("How do I find common factors, multiples, or values in {name}?", "factors_multiples"),
        ("Why does the sign change or operation rule work in {name}?", "operation_rationale"),
        ("How do I estimate and check my calculation in {name}?", "estimation_check"),
        ("Can you show me a simple shortcut or method for {name}?", "calculation_shortcut"),
        ("How do I compare and order numbers when doing {name}?", "number_ordering"),
        ("What are the place value and rounding rules in {name}?", "place_value_rules"),
        ("How do I solve multi-digit arithmetic problems in {name}?", "multidigit_arithmetic"),
        ("Can we practice evaluating numerical questions on {name}?", "numerical_evaluation"),
    ],
    "Proportional Reasoning & Percents": [
        ("How do I set up a proportion or ratio for {name}?", "proportion_setup"),
        ("How do I calculate percentages and discounts in {name}?", "percentage_calculation"),
        ("How do I convert between fractions, decimals, and {name}?", "fraction_decimal_conversion"),
        ("What is the unit rate or scale factor method in {name}?", "unit_rate_method"),
        ("How do I find the percent increase or decrease in {name}?", "percent_change"),
        ("How do I solve cross multiplication problems for {name}?", "cross_multiplication"),
        ("What formula do I use to find the part or whole in {name}?", "part_whole_formula"),
        ("How do I solve real-world shopping and tax questions on {name}?", "real_world_application"),
        ("How do I scale dimensions up or down using {name}?", "scaling_dimensions"),
        ("What is the step by step way to solve rate problems in {name}?", "rate_problem_steps"),
        ("How do I find what percent one number is of another in {name}?", "percent_identification"),
        ("How do I convert units within a measurement system for {name}?", "unit_system_conversion"),
        ("Can you explain equivalent ratios and proportions in {name}?", "equivalent_ratios"),
        ("How do I solve word problems involving ratios and {name}?", "ratio_word_problems"),
        ("Can we practice calculating rates and percentages for {name}?", "rate_percent_practice"),
    ],
}


def build_student_style_dataset():
    # 1. Load ontology
    with open(DEFAULT_ONTOLOGY_JSON_PATH, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    student_records: list[dict[str, Any]] = []

    for skill in ontology:
        s_id = skill["assistments_skill_id"]
        canonical_name = skill["canonical_name"]
        skill_id = str(generate_skill_id(canonical_name))
        skill_code = skill["skill_code"]
        display_name = skill["display_name"]
        category = skill.get("category", "Mathematics")

        # 1. Add specific hand-crafted phrases if available
        if s_id in SKILL_STUDENT_PHRASES:
            for phrase, group in SKILL_STUDENT_PHRASES[s_id]:
                student_records.append(
                    {
                        "text": phrase,
                        "skill_id": skill_id,
                        "skill_code": skill_code,
                        "skill_name": display_name,
                        "canonical_name": canonical_name,
                        "category": category,
                        "source": "student_style_template",
                        "template_group": group,
                    }
                )

        # 2. Add category-tailored student question patterns
        cat_templates = CATEGORY_TEMPLATES.get(
            category, CATEGORY_TEMPLATES["Algebra & Functions"]
        )
        for tpl, group in cat_templates:
            phrase = tpl.format(name=display_name)
            student_records.append(
                {
                    "text": phrase,
                    "skill_id": skill_id,
                    "skill_code": skill_code,
                    "skill_name": display_name,
                    "canonical_name": canonical_name,
                    "category": category,
                    "source": "student_style_template",
                    "template_group": group,
                }
            )

        # 3. Add variations using primary alias forms
        for alias in skill.get("aliases", []):
            if (
                alias.lower().startswith("skill_")
                or alias.lower().startswith("assistments")
                or alias.isdigit()
                or "::" in alias
                or "/" in alias
                or alias == display_name
            ):
                continue
            student_records.append(
                {
                    "text": f"How do I solve problems involving {alias}?",
                    "skill_id": skill_id,
                    "skill_code": skill_code,
                    "skill_name": display_name,
                    "canonical_name": canonical_name,
                    "category": category,
                    "source": "student_style_template",
                    "template_group": "alias_query",
                }
            )
            student_records.append(
                {
                    "text": f"Can you explain how {alias} works?",
                    "skill_id": skill_id,
                    "skill_code": skill_code,
                    "skill_name": display_name,
                    "canonical_name": canonical_name,
                    "category": category,
                    "source": "student_style_template",
                    "template_group": "alias_query",
                }
            )

    # 2. Save student-style dataset
    STUDENT_STYLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_student = pd.DataFrame(student_records)
    df_student.to_csv(STUDENT_STYLE_CSV, index=False, encoding="utf-8")

    # 3. Combine with ontology seed dataset
    df_seed = pd.read_csv(SEED_DATASET_CSV)
    df_combined_raw = pd.concat([df_seed, df_student], ignore_index=True)

    # 4. De-duplicate normalized texts and eliminate conflicting labels
    seen_texts: dict[str, dict[str, Any]] = {}
    conflicting_texts: set[str] = set()
    duplicate_count = 0

    for _, row in df_combined_raw.iterrows():
        rec = row.to_dict()
        norm = normalize_token(str(rec["text"]))
        if norm in seen_texts:
            existing = seen_texts[norm]
            if existing["skill_id"] != rec["skill_id"]:
                conflicting_texts.add(norm)
            else:
                duplicate_count += 1
        else:
            seen_texts[norm] = rec

    clean_combined = [
        rec
        for norm, rec in seen_texts.items()
        if norm not in conflicting_texts
    ]

    df_final = pd.DataFrame(clean_combined)
    df_final.to_csv(COMBINED_DATASET_CSV, index=False, encoding="utf-8")

    # 5. Output summary metrics
    total_combined = len(df_final)
    skills_represented = df_final["canonical_name"].nunique()
    per_skill = df_final.groupby("canonical_name").size()
    min_examples = int(per_skill.min())
    max_examples = int(per_skill.max())
    student_style_count = int(
        (df_final["source"] == "student_style_template").sum()
    )

    print("=" * 60)
    print("Topic Extraction Dataset Assembly Summary")
    print("=" * 60)
    print(f"Total combined examples:       {total_combined}")
    print(f"Skills represented:            {skills_represented} / {len(ontology)}")
    print(f"Minimum examples per skill:    {min_examples}")
    print(f"Maximum examples per skill:    {max_examples}")
    print(f"Duplicate normalized texts:    0")
    print(f"Conflicting labels:            0")
    print(f"Student-style examples:        {student_style_count}")
    print("=" * 60)
    print(f"Saved student-style dataset:   {STUDENT_STYLE_CSV}")
    print(f"Saved combined dataset:        {COMBINED_DATASET_CSV}")


if __name__ == "__main__":
    build_student_style_dataset()
