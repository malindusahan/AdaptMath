"""Prepare the preregistered MD7-R2-TELL-C1 blinded review extension.

The 128 authored states are frozen and hashed before the frozen inference
implementation is imported or either model is scored. This script never
trains, promotes, calls Tutor APIs, or reads protected final-test material.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "pedagogical-move-selection" / "results"
REVIEW1 = RESULTS / "md7_r2_tell_c1_blinded_review_v1"
OLD_DIAGNOSTIC = RESULTS / "md7_telling_real_state_diagnostic_v1"
PREPARATION = RESULTS / "md7_r2_tell_c1_preparation_v1"
BASELINE = ROOT / "pedagogical-move-selection" / "models" / "candidates" / "md7r1_epoch3"
CANDIDATE = ROOT / "pedagogical-move-selection" / "models" / "candidates" / "md7_r2_tell_c1" / "epoch2"
V1_PREPARE = REVIEW1 / "prepare_review.py"

MOVES = ("generic", "probing", "focus", "telling")
STRATA = (
    "A_first_clear_incorrect_answer",
    "B_first_confusion_or_first_i_dont_know",
    "C_recoverable_after_one_meaningful_scaffold",
    "D_repeated_misconception_after_multiple_scaffolds",
    "E_repeated_confusion_after_multiple_scaffolds",
    "F_direct_explanation_request_after_previous_support",
    "G_correct_progress_after_previous_difficulty",
    "H_partial_recovering_progress_after_previous_difficulty",
)
REVIEW_SET_SEED = 2026082901
MODEL_BLINDING_SEED = 2026082902
SIMILARITY_THRESHOLD = 0.80
MIN_MODEL_DISAGREEMENTS = 8
MIN_DECISIVE_DISAGREEMENTS = 8
EXPECTED_V1_ANALYSIS_HASH = "adb98cf8c978479d7aedb437c01bbbbe62e13b6bebd94acf5a932f0397dafd2c"
EXPECTED_BASELINE_WEIGHT = "d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13"
EXPECTED_CANDIDATE_WEIGHT = "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"


def c(
    draft_id: str,
    stratum_index: int,
    problem: str,
    history: Sequence[tuple[str, str]],
    latest_student_state: str,
) -> dict[str, Any]:
    return {
        "draft_id": draft_id,
        "semantic_stratum": STRATA[stratum_index],
        "problem": problem,
        "dialogue_history": [{"user": user, "text": text} for user, text in history],
        "latest_student_state": latest_student_state,
    }


# Independently authored before model inference. There are no expected labels.
DRAFT_CASES: list[dict[str, Any]] = [
    c("A01", 0, "Simplify 3(2x + 5) - 4.",
      [("teacher", "What expression do you get after simplifying?"),
       ("student", "I multiplied 3 by 2x, then subtracted 4, so I got 6x + 1.")],
      "The learner's first answer fails to distribute to the constant term."),
    c("A02", 0, "Find two thirds of 45.",
      [("teacher", "How would you calculate the fraction of the amount?"),
       ("student", "I divided 45 by 2 and then multiplied by 3, giving 67.5.")],
      "The first attempt reverses the numerator and denominator operations."),
    c("A03", 0, "The temperature rises from -4 degrees Celsius to 7 degrees Celsius. By how many degrees did it rise?",
      [("teacher", "What change in temperature do you obtain?"),
       ("student", "I subtracted 4 from 7, so the rise is 3 degrees.")],
      "The first response ignores that the starting temperature is negative."),
    c("A04", 0, "A rectangle is 9 cm long and 4 cm wide. Find its perimeter.",
      [("teacher", "What calculation represents the perimeter?"),
       ("student", "I multiplied 9 by 4 and got 36 centimetres.")],
      "The first answer calculates area instead of perimeter."),
    c("A05", 0, "Find the mean of 5, 8, 12, and 15.",
      [("teacher", "Show your calculation for the mean."),
       ("student", "The middle numbers are 8 and 12, so the mean is 10.")],
      "The learner initially uses the median procedure for a mean."),
    c("A06", 0, "Write 0.35 as a percentage.",
      [("teacher", "What percentage is equivalent to the decimal?"),
       ("student", "I divided by 100 and got 0.0035 percent.")],
      "The first conversion applies the scale factor in the wrong direction."),
    c("A07", 0, "Solve 5x + 2 = 27.",
      [("teacher", "What value do you get for x?"),
       ("student", "I divided 27 by 5 first and then subtracted 2, so x is 3.4.")],
      "The first solution undoes the operations in an invalid order."),
    c("A08", 0, "One angle on a straight line is 112 degrees. Find the adjacent angle.",
      [("teacher", "How large is the adjacent angle?"),
       ("student", "I subtracted from 360 and got 248 degrees.")],
      "The first answer uses the full-turn total instead of a straight-line total."),
    c("A09", 0, "A fair six-sided die is rolled once. Find the probability of not rolling a 6.",
      [("teacher", "What probability did you find?"),
       ("student", "There is one number 6, so the probability is 1/6.")],
      "The learner gives the event probability rather than its complement."),
    c("A10", 0, "Translate the point (-2, 5) by the vector (4, -3).",
      [("teacher", "Where does the point move?"),
       ("student", "I multiplied the coordinates and got (-8, -15).")],
      "The first response multiplies coordinates instead of adding the vector."),
    c("A11", 0, "Simplify 10^3 multiplied by 10^2.",
      [("teacher", "How would you combine the powers?"),
       ("student", "I multiplied the exponents, so my answer is 10^6.")],
      "The learner first multiplies exponents for a product of equal bases."),
    c("A12", 0, "Simplify the ratio 18:30.",
      [("teacher", "What simplified ratio do you get?"),
       ("student", "I subtracted 18 from 30, so I wrote 18:12.")],
      "The first answer uses subtraction rather than a common divisor."),
    c("A13", 0, "A parallelogram has base 13 cm and perpendicular height 7 cm. Find its area.",
      [("teacher", "What area do you calculate?"),
       ("student", "I used one half times 13 times 7 and got 45.5 square centimetres.")],
      "The first response incorrectly applies the triangle factor of one half."),
    c("A14", 0, "Calculate 1.2 + 0.08.",
      [("teacher", "What sum do you obtain?"),
       ("student", "I joined the digits and got 1.208.")],
      "The learner's first decimal addition does not align place values."),
    c("A15", 0, "The sequence is 4, 9, 14, 19, ... Find the next term.",
      [("teacher", "What comes next in the sequence?"),
       ("student", "The differences are 5, so I doubled 19 and got 38.")],
      "The first answer identifies but does not apply the constant difference."),
    c("A16", 0, "For y = 3x - 2, find y when x = 4.",
      [("teacher", "What value of y do you get?"),
       ("student", "I used 3 + 4 - 2, so y equals 5.")],
      "The first substitution treats multiplication as addition."),

    c("B01", 1, "Calculate 3/5 divided by 2/7.",
      [("teacher", "How would you begin dividing the fractions?"),
       ("student", "I don't know which fraction, if any, should be turned over.")],
      "The learner expresses first uncertainty about fraction division."),
    c("B02", 1, "Find the lowest common multiple of 18 and 24.",
      [("teacher", "What approach could you use to find the lowest common multiple?"),
       ("student", "I am not sure how a common multiple is different from a common factor.")],
      "The learner reports a first conceptual distinction they cannot recall."),
    c("B03", 1, "Solve 2(3x - 1) = 16.",
      [("teacher", "What could you do first with the bracketed equation?"),
       ("student", "I don't know whether I should expand the bracket or divide first.")],
      "The learner has first-time procedural uncertainty with two possible routes."),
    c("B04", 1, "A regular polygon has an exterior angle of 24 degrees. Find the number of sides.",
      [("teacher", "What relationship connects exterior angles and the number of sides?"),
       ("student", "I can't remember what the exterior angles add up to.")],
      "The learner has an initial recall gap before receiving support."),
    c("B05", 1, "A circle has radius 10 cm. Find the length of a 72-degree arc in terms of pi.",
      [("teacher", "How would you start finding only part of the circumference?"),
       ("student", "I don't know how the 72 degrees fits into the calculation.")],
      "The learner first expresses uncertainty about the fractional-circle relationship."),
    c("B06", 1, "An investment of $500 earns 6% compound interest each year. Write an expression for its value after 4 years.",
      [("teacher", "How can you represent four repeated percentage increases?"),
       ("student", "I have no idea what number should be raised to the fourth power.")],
      "The learner states an initial confusion about the compound-growth factor."),
    c("B07", 1, "A data set has lower quartile 14 and upper quartile 31. Find its interquartile range.",
      [("teacher", "What would you do with the two quartiles?"),
       ("student", "I don't remember how the interquartile range is calculated.")],
      "The learner reports a first recall failure for a single statistic."),
    c("B08", 1, "Expand (x + 4)(x - 2).",
      [("teacher", "How would you begin multiplying the two brackets?"),
       ("student", "I am unsure which terms need to multiply each other.")],
      "The learner expresses first confusion before any multiplication scaffold."),
    c("B09", 1, "Write the product (3 × 10^5)(2 × 10^-3) in scientific notation.",
      [("teacher", "How might you combine the coefficients and powers of ten?"),
       ("student", "I don't know what to do when one exponent is negative.")],
      "The learner first signals uncertainty about exponent combination."),
    c("B10", 1, "The cost of apples is directly proportional to their mass. Three kilograms cost $7.20. Find the cost of five kilograms.",
      [("teacher", "How could you use the cost for three kilograms?"),
       ("student", "I don't know how to scale from three kilograms to five.")],
      "The learner gives a first statement of uncertainty about direct proportion."),
    c("B11", 1, "Rearrange v = u + at to make a the subject.",
      [("teacher", "What operation could begin isolating a?"),
       ("student", "I am not sure which term should be moved first.")],
      "The learner has first procedural uncertainty when rearranging a formula."),
    c("B12", 1, "A right triangle has legs 7 cm and 11 cm. Find its hypotenuse.",
      [("teacher", "How would you set up the calculation?"),
       ("student", "I know Pythagoras is involved, but I don't know which values to square.")],
      "The learner first requests orientation to the theorem."),
    c("B13", 1, "A fair coin is tossed three times. Find the probability of at least one head.",
      [("teacher", "What cases could help you calculate at least one head?"),
       ("student", "I don't know how many different outcomes I need to count.")],
      "The learner expresses initial uncertainty about the event structure."),
    c("B14", 1, "State the gradient and y-intercept of 4y = 12x - 20.",
      [("teacher", "How could you put the equation into a familiar line form?"),
       ("student", "I don't know how to read the gradient while y has a coefficient of 4.")],
      "The learner identifies a first obstacle in interpreting the equation."),
    c("B15", 1, "Find the volume of a cylinder with radius 3 cm and height 8 cm in terms of pi.",
      [("teacher", "What measurements belong in the cylinder volume formula?"),
       ("student", "I can't remember whether the radius is squared or the height is squared.")],
      "The learner has an initial formula recall gap."),
    c("B16", 1, "Find the nth term of 7, 12, 17, 22, ...",
      [("teacher", "How would you turn the pattern into an nth-term rule?"),
       ("student", "I can see it increases by 5, but I don't know how that becomes a formula using n.")],
      "The learner first expresses uncertainty moving from difference to general rule."),

    c("C01", 2, "A $160 tablet is reduced by 30%. Find the sale price.",
      [("teacher", "What sale price did you initially find?"),
       ("student", "I subtracted 30 and got $130."),
       ("teacher", "First calculate 30 percent of 160, then subtract that discount from 160."),
       ("student", "Thirty percent is 0.30 times 160, which is 48, so I should do 160 minus 48.")],
      "One scaffold produces the correct discount amount and final operation."),
    c("C02", 2, "Calculate (-6)(-7).",
      [("teacher", "What answer did you get?"),
       ("student", "I wrote -42 because both numbers have minus signs."),
       ("teacher", "Recall the sign rule when two negative factors are multiplied."),
       ("student", "Two negatives give a positive product, so the answer should be 42.")],
      "The learner corrects the sign after one concise scaffold."),
    c("C03", 2, "Calculate 5/8 + 1/6.",
      [("teacher", "How did you try to add the fractions?"),
       ("student", "I added top and bottom and got 6/14."),
       ("teacher", "Rename both fractions using a denominator divisible by both 8 and 6."),
       ("student", "A common denominator is 24, so I get 15/24 plus 4/24.")],
      "One denominator cue leads to correct equivalent fractions."),
    c("C04", 2, "Find the gradient between the points (1, -2) and (7, 10).",
      [("teacher", "What gradient did you first calculate?"),
       ("student", "I added coordinates and wrote 8/8 = 1."),
       ("teacher", "Use change in y divided by change in x, keeping the point order consistent."),
       ("student", "The changes are 10 - (-2) = 12 and 7 - 1 = 6, so the gradient is 12/6.")],
      "After one scaffold the learner forms the correct differences."),
    c("C05", 2, "A recipe uses flour and sugar in the ratio 7:3. If 21 cups of flour are used, how much sugar is needed?",
      [("teacher", "What did you do with the ratio?"),
       ("student", "I added 7 and 3 to 21 and got 31."),
       ("teacher", "Find the multiplier that changes 7 parts of flour into 21 cups."),
       ("student", "The multiplier is 3, so the sugar amount should be 3 times 3 cups.")],
      "One scale-factor prompt produces the correct multiplicative structure."),
    c("C06", 2, "A card is chosen from a standard 52-card deck. Find the probability that it is not a heart.",
      [("teacher", "What probability did you write first?"),
       ("student", "There are 13 hearts, so I wrote 13/52."),
       ("teacher", "Use the complement of drawing a heart."),
       ("student", "The complement has 52 - 13 = 39 cards, so it is 39/52.")],
      "The learner applies one complement cue correctly."),
    c("C07", 2, "An L-shaped floor is formed from a 10 m by 8 m rectangle with a 3 m by 2 m corner removed. Find its area.",
      [("teacher", "What area did you initially calculate?"),
       ("student", "I added all four dimensions and got 23 square metres."),
       ("teacher", "Find the large rectangle's area, then subtract the removed corner's area."),
       ("student", "The large area is 80 and the missing area is 6, so I need 80 minus 6.")],
      "A single decomposition scaffold yields the correct area structure."),
    c("C08", 2, "Simplify p^9 / p^4 for nonzero p.",
      [("teacher", "What did you first do with the exponents?"),
       ("student", "I divided 9 by 4."),
       ("teacher", "For equal bases in a quotient, subtract the denominator exponent from the numerator exponent."),
       ("student", "Then the exponent is 9 - 4 = 5, so the result is p^5.")],
      "One rule reminder leads to a complete corrected result."),
    c("C09", 2, "Solve 3x - 8 < 13.",
      [("teacher", "What inequality did you get at first?"),
       ("student", "I divided 13 by 3 and then subtracted 8."),
       ("teacher", "Undo the subtraction of 8 before dividing by 3."),
       ("student", "Adding 8 gives 3x < 21, and then I divide both sides by 3.")],
      "One inverse-operation scaffold produces the correct intermediate inequality."),
    c("C10", 2, "Values 2, 5, and 8 occur with frequencies 3, 2, and 1. Find the mean.",
      [("teacher", "How did you first use the frequencies?"),
       ("student", "I ignored them and averaged 2, 5, and 8."),
       ("teacher", "Multiply each value by its frequency, then divide the total by the sum of frequencies."),
       ("student", "The weighted total is 2(3) + 5(2) + 8(1), and the total frequency is 6.")],
      "One weighted-mean scaffold yields a correct calculation setup."),
    c("C11", 2, "Calculate 4.95 divided by 0.5.",
      [("teacher", "How did you begin?"),
       ("student", "I removed the decimal from 0.5 and used 4.95 divided by 5."),
       ("teacher", "Multiply both dividend and divisor by 10 to keep the quotient equivalent."),
       ("student", "That gives 49.5 divided by 5, so I can continue from there.")],
      "The learner correctly rescales both numbers after one scaffold."),
    c("C12", 2, "Factor x^2 + 9x + 20.",
      [("teacher", "Which numbers did you first try?"),
       ("student", "I used 2 and 10 because they multiply to 20."),
       ("teacher", "The pair must multiply to 20 and also add to the middle coefficient 9."),
       ("student", "Four and five meet both conditions, so the brackets should use 4 and 5.")],
      "One two-condition reminder leads to the correct factor pair."),
    c("C13", 2, "Convert 3.6 kilograms to grams.",
      [("teacher", "What conversion did you first make?"),
       ("student", "I multiplied by 100 and got 360 grams."),
       ("teacher", "A kilogram contains 1,000 grams."),
       ("student", "Then I multiply 3.6 by 1,000 and get 3,600 grams.")],
      "One unit-size reminder produces the correct conversion."),
    c("C14", 2, "Two parallel lines are cut by a transversal. An alternate interior angle is 68 degrees. Find its alternate interior partner.",
      [("teacher", "What did you first think the partner angle was?"),
       ("student", "I subtracted from 180 and got 112 degrees."),
       ("teacher", "Alternate interior angles between parallel lines are equal."),
       ("student", "Then the partner angle is also 68 degrees.")],
      "The learner applies one angle-rule cue correctly."),
    c("C15", 2, "For f(x) = 2x^2 - 1, find f(3).",
      [("teacher", "What did you calculate first?"),
       ("student", "I used 2 times 3 and then squared the result."),
       ("teacher", "Substitute 3 for x inside x squared before multiplying by 2."),
       ("student", "So I use 2(3^2) - 1, which is 18 - 1.")],
      "One operation-order scaffold produces the correct substitution."),
    c("C16", 2, "A circle has diameter 18 cm. Find its circumference in terms of pi.",
      [("teacher", "What expression did you write first?"),
       ("student", "I used 2 pi times 18."),
       ("teacher", "In C = pi d, the given diameter can be used directly."),
       ("student", "Then the circumference is pi times 18, or 18pi centimetres.")],
      "One formula distinction leads to the correct circumference."),

    c("D01", 3, "Solve 6x - 5 = 25.",
      [("teacher", "What did you do first?"),
       ("student", "I divided 25 by 6 and then subtracted 5."),
       ("teacher", "Keep the equation balanced by undoing minus 5 on both sides first."),
       ("student", "I still divided first because 6 is next to x."),
       ("teacher", "Write 6x - 5 + 5 = 25 + 5 before any division."),
       ("student", "I think I should divide 25 by 6 first anyway.")],
      "The inverse-operation order misconception persists after two scaffolds."),
    c("D02", 3, "Calculate -9 + 14.",
      [("teacher", "What answer did you get?"),
       ("student", "I added 9 and 14 and kept the minus sign, giving -23."),
       ("teacher", "The numbers have different signs, so compare their magnitudes and subtract."),
       ("student", "I still add them because the symbol says plus."),
       ("teacher", "On a number line, begin at -9 and move 14 units to the right."),
       ("student", "I still think plus means the answer must be -23.")],
      "The signed-number misconception repeats after arithmetic and visual scaffolds."),
    c("D03", 3, "Calculate 2/5 + 3/7.",
      [("teacher", "How did you add the fractions?"),
       ("student", "I added the numerators and denominators to get 5/12."),
       ("teacher", "Fractions need equal-sized parts, so first use a common denominator."),
       ("student", "I still get 12 by adding 5 and 7."),
       ("teacher", "Use 35 as a denominator and rename both fractions."),
       ("student", "I prefer 5/12 because I added straight across.")],
      "The add-across misconception persists after two denominator prompts."),
    c("D04", 3, "A salary of $900 increases by 8%. Find the new salary.",
      [("teacher", "What new salary did you calculate?"),
       ("student", "I added 8 dollars and got $908."),
       ("teacher", "Eight percent means eight hundredths of 900, not eight dollars."),
       ("student", "I still think an increase by 8 means add 8."),
       ("teacher", "Calculate 0.08 times 900 as the increase before adding it."),
       ("student", "My answer remains $908.")],
      "A fixed-amount interpretation of percentage persists after two supports."),
    c("D05", 3, "A square has side length 12 cm. Find its perimeter.",
      [("teacher", "What did you calculate?"),
       ("student", "I squared 12 and got 144 centimetres."),
       ("teacher", "Squaring the side gives area; perimeter totals the four side lengths."),
       ("student", "A square still makes me think I should square 12."),
       ("teacher", "Write 12 + 12 + 12 + 12 around the boundary."),
       ("student", "I still choose 144.")],
      "The area-perimeter confusion remains after conceptual and explicit scaffolds."),
    c("D06", 3, "Find the slope through (2, 3) and (8, 15).",
      [("teacher", "How did you find the slope?"),
       ("student", "I added the y-values over the x-values and got 18/10."),
       ("teacher", "Slope uses the change in y over the change in x."),
       ("student", "I used 15 + 3 over 8 + 2 again."),
       ("teacher", "Write (15 - 3)/(8 - 2) using matching point order."),
       ("student", "I still think adding the coordinates is correct.")],
      "The coordinate-sum misconception persists after two slope scaffolds."),
    c("D07", 3, "Simplify q^4 multiplied by q^6.",
      [("teacher", "How did you combine the powers?"),
       ("student", "I multiplied 4 by 6 and got q^24."),
       ("teacher", "Count the total repeated q factors in the product."),
       ("student", "I still multiply the exponent numbers."),
       ("teacher", "Expand four q factors followed by six more q factors and count them."),
       ("student", "I keep getting 24 because it is multiplication.")],
      "The exponent-product misconception repeats after two representations."),
    c("D08", 3, "A fair coin is tossed twice. Find the probability of two heads.",
      [("teacher", "How did you combine the two tosses?"),
       ("student", "I added 1/2 and 1/2 to get 1."),
       ("teacher", "Both heads must occur together, so combine the independent stages multiplicatively."),
       ("student", "There are two tosses, so I still add the probabilities."),
       ("teacher", "Use the HH path: 1/2 for the first stage and 1/2 for the second stage."),
       ("student", "I still say the answer is 1.")],
      "The learner continues adding sequential probabilities after two prompts."),
    c("D09", 3, "Evaluate 24 / 3 + 2 × 5.",
      [("teacher", "What order did you use?"),
       ("student", "I worked left to right after adding 3 + 2, and got 24/5 times 5 = 24."),
       ("teacher", "Complete division and multiplication before addition."),
       ("student", "I still added 3 and 2 first because they are in the middle."),
       ("teacher", "Separate the expression into 24/3 and 2×5, evaluate each, then add."),
       ("student", "I keep combining 3 + 2 first.")],
      "The operation-order error persists after rule and chunking scaffolds."),
    c("D10", 3, "Two similar rectangles have corresponding widths 6 cm and 15 cm. The smaller length is 10 cm. Find the larger length.",
      [("teacher", "How did you scale the length?"),
       ("student", "I added the width difference 9 to 10 and got 19."),
       ("teacher", "Similar figures use one multiplicative scale factor for every corresponding length."),
       ("student", "I still add 9 because that changed 6 into 15."),
       ("teacher", "Find 15 divided by 6 and multiply 10 by that factor."),
       ("student", "I think 19 is still the matching length.")],
      "The additive scale misconception persists after two supports."),
    c("D11", 3, "Find the median of 3, 6, 8, 20, and 28.",
      [("teacher", "What median did you find?"),
       ("student", "I added the values and divided by five."),
       ("teacher", "That computes the mean; the median is the ordered middle value."),
       ("student", "I still divide the total because median means average to me."),
       ("teacher", "Cross off equal numbers from each end until one middle number remains."),
       ("student", "I still want to calculate the total divided by five.")],
      "The mean-median misconception remains after definition and procedure prompts."),
    c("D12", 3, "Solve -5x ≤ 35.",
      [("teacher", "What inequality did you obtain?"),
       ("student", "I divided by -5 and wrote x ≤ -7."),
       ("teacher", "Dividing by a negative reverses the inequality direction."),
       ("student", "I do not think the symbol should change."),
       ("teacher", "Test a value such as x = -8 in the original inequality to check the direction."),
       ("student", "I still keep x ≤ -7.")],
      "The inequality-direction misconception persists after rule and checking scaffolds."),
    c("D13", 3, "Reflect the point (-4, 6) across the x-axis.",
      [("teacher", "What image point did you get?"),
       ("student", "I changed both signs and got (4, -6)."),
       ("teacher", "Across the x-axis, horizontal position stays fixed while vertical position changes sign."),
       ("student", "I still change x because it is called the x-axis."),
       ("teacher", "Keep x = -4 and mirror only the distance above the axis to below it."),
       ("student", "I still think the answer begins with positive 4.")],
      "The learner continues changing both coordinates after two geometric cues."),
    c("D14", 3, "Expand 5(2y - 7).",
      [("teacher", "What expression did you obtain?"),
       ("student", "I got 10y - 7."),
       ("teacher", "The factor outside the bracket multiplies every term inside."),
       ("student", "I multiplied the y term, so I think that is enough."),
       ("teacher", "Write both products: 5×2y and 5×(-7)."),
       ("student", "I still leave the constant as -7.")],
      "The incomplete-distribution error remains after two explicit prompts."),
    c("D15", 3, "Use the quadratic formula for x^2 + 4x - 5 = 0.",
      [("teacher", "Which coefficient values did you identify?"),
       ("student", "I used a = 1, b = -4, and c = 5."),
       ("teacher", "Copy each coefficient with the sign shown in the equation."),
       ("student", "I still switch both signs when I put them into the formula."),
       ("teacher", "Write b = 4 and c = -5 before substituting anything."),
       ("student", "I think b should be -4 because the formula starts with -b.")],
      "Coefficient signs remain conflated with formula operations after two scaffolds."),
    c("D16", 3, "A block has mass 240 g and volume 30 cm^3. Find its density.",
      [("teacher", "What density calculation did you use?"),
       ("student", "I multiplied 240 by 30."),
       ("teacher", "Density is mass per unit volume, so divide mass by volume."),
       ("student", "I still multiply because there are two measurements."),
       ("teacher", "Write 240 grams shared across 30 cubic centimetres as 240/30."),
       ("student", "I still think the operation is multiplication.")],
      "The density-operation misconception persists after formula and unit-rate supports."),

    c("E01", 4, "Solve y = x + 4 and 2x + y = 19.",
      [("teacher", "How can the expression for y be used in the other equation?"),
       ("student", "I don't understand how the equations connect."),
       ("teacher", "Replace y in 2x + y = 19 with x + 4."),
       ("student", "I can see x + 4, but I don't know where it belongs."),
       ("teacher", "Write 2x + (x + 4) = 19 and collect the x terms."),
       ("student", "I am still lost about what to do with that new equation.")],
      "The learner remains confused after substitution is explicitly constructed."),
    c("E02", 4, "Calculate 7.56 divided by 0.12.",
      [("teacher", "How could you remove the decimal from the divisor?"),
       ("student", "I don't know how many places to move it."),
       ("teacher", "Multiply both numbers by 100, giving 756 divided by 12."),
       ("student", "I understand the new division but cannot work it out."),
       ("teacher", "Ask how many groups of 12 make 720, then handle the remaining 36."),
       ("student", "I still cannot combine those parts into the quotient.")],
      "Repeated confusion persists after the decimal and division steps are narrowed."),
    c("E03", 4, "Rearrange P = 2l + 2w to make w the subject.",
      [("teacher", "Which term needs to be isolated?"),
       ("student", "I do not know how to move the 2l."),
       ("teacher", "Subtract 2l from both sides first."),
       ("student", "Then I have P - 2l, but I don't know what happens next."),
       ("teacher", "The remaining right side is 2w, so divide the entire left side by 2."),
       ("student", "I still cannot write the final formula for w.")],
      "The learner remains unable to finish after two explicit inverse-operation steps."),
    c("E04", 4, "A bag has 4 red and 6 blue counters. Two are drawn without replacement. Find the probability the second is red given the first was blue.",
      [("teacher", "What remains in the bag after a blue counter is removed?"),
       ("student", "I don't know how the condition changes the counts."),
       ("teacher", "There are still 4 red counters, but only 9 counters total."),
       ("student", "I see the numbers 4 and 9, but I don't know how to use them."),
       ("teacher", "Conditional on the first blue, form favorable remaining red over all remaining counters."),
       ("student", "I still cannot decide what probability to write.")],
      "The learner remains confused after the conditional sample space is fully specified."),
    c("E05", 4, "Write (8 × 10^7) / (2 × 10^3) in scientific notation.",
      [("teacher", "How could you handle the coefficients and powers separately?"),
       ("student", "I don't know whether the exponents add or subtract."),
       ("teacher", "Divide 8 by 2, and subtract the denominator exponent from the numerator exponent."),
       ("student", "I got the coefficient 4, but I don't know the power of ten."),
       ("teacher", "Compute 7 - 3 for the exponent."),
       ("student", "I am still unsure how to write the final scientific notation.")],
      "Confusion persists after both components of the calculation are supplied."),
    c("E06", 4, "For a fixed journey, travel time is inversely proportional to speed. It takes 6 hours at 50 km/h. Find the time at 75 km/h.",
      [("teacher", "What stays constant for this fixed journey?"),
       ("student", "I don't know how inverse proportion works here."),
       ("teacher", "The product speed times time stays constant, so first calculate 50 times 6."),
       ("student", "That gives 300, but I don't know what 300 represents."),
       ("teacher", "Set 75 times the new time equal to 300."),
       ("student", "I still cannot see how to obtain the new time.")],
      "The learner remains confused after the invariant and equation are given."),
    c("E07", 4, "In a triangle, two sides are 8 cm and 11 cm with included angle 60 degrees. Find the third side.",
      [("teacher", "Which rule uses two sides and their included angle?"),
       ("student", "I don't know which triangle formula applies."),
       ("teacher", "Use the cosine rule c^2 = a^2 + b^2 - 2ab cos C."),
       ("student", "I can copy the formula, but I don't know where 60 goes."),
       ("teacher", "Substitute C = 60 degrees and a, b as 8 and 11."),
       ("student", "I still cannot organize the substitution.")],
      "The learner remains stuck after the applicable rule and variable roles are supplied."),
    c("E08", 4, "A cumulative-frequency graph represents 80 observations. Explain how to estimate the median from the graph.",
      [("teacher", "Which cumulative frequency corresponds to the median?"),
       ("student", "I don't know what point to look for."),
       ("teacher", "The median is halfway through 80 observations, so locate cumulative frequency 40."),
       ("student", "I found 40 on the vertical axis but don't know what to do next."),
       ("teacher", "Move horizontally to the curve and then vertically down to read the data value."),
       ("student", "I still don't understand how that gives the median.")],
      "Repeated confusion remains after the graph-reading path is specified."),
    c("E09", 4, "Simplify 3/x + 2/(x + 1).",
      [("teacher", "What denominator could combine the two algebraic fractions?"),
       ("student", "I don't know how to make their denominators match."),
       ("teacher", "Use x(x + 1) as a common denominator."),
       ("student", "I can write that denominator, but I don't know how each numerator changes."),
       ("teacher", "Multiply the first numerator by x + 1 and the second numerator by x."),
       ("student", "I am still unable to form the combined numerator.")],
      "The learner remains confused after common denominator and numerator multipliers are given."),
    c("E10", 4, "Solve 3x^2 + 2x - 4 = 0 using the quadratic formula.",
      [("teacher", "What values are a, b, and c?"),
       ("student", "I know a is 3, but I don't know how to use the other coefficients."),
       ("teacher", "Use a = 3, b = 2, and c = -4 in the formula."),
       ("student", "I wrote the values down, but the discriminant still confuses me."),
       ("teacher", "Calculate b^2 - 4ac as 2^2 - 4(3)(-4)."),
       ("student", "I still cannot work out what belongs under the square root.")],
      "Confusion persists after coefficients and discriminant substitution are explicit."),
    c("E11", 4, "A cone has radius 5 cm and perpendicular height 12 cm. Find its volume in terms of pi.",
      [("teacher", "Which measurements enter the cone volume formula?"),
       ("student", "I don't remember the formula."),
       ("teacher", "Use one third times pi times radius squared times height."),
       ("student", "I know the formula now but don't know which number is squared."),
       ("teacher", "Square the radius 5, then multiply by height 12 and divide by 3."),
       ("student", "I still cannot assemble the calculation.")],
      "The learner remains confused after formula and operation roles are explained."),
    c("E12", 4, "Rotate the point (3, -1) by 90 degrees anticlockwise about the origin.",
      [("teacher", "What coordinate rule could represent this rotation?"),
       ("student", "I don't know what happens to the coordinates."),
       ("teacher", "For 90 degrees anticlockwise, map (x, y) to (-y, x)."),
       ("student", "I can see the rule but don't know how to substitute 3 and -1."),
       ("teacher", "Place y = -1 into -y and x = 3 into the second coordinate."),
       ("student", "I still cannot decide the signs of the image point.")],
      "Repeated sign confusion remains after the transformation rule is instantiated."),
    c("E13", 4, "Convert the recurring decimal 0.272727... to a fraction.",
      [("teacher", "How many repeating digits are there?"),
       ("student", "There are two, but I don't know how that helps."),
       ("teacher", "Let x = 0.272727... and multiply by 100 so the repeating parts align."),
       ("student", "I have 100x = 27.2727..., but I don't know the next step."),
       ("teacher", "Subtract the original x equation from the 100x equation."),
       ("student", "I still don't understand how subtraction removes the decimals.")],
      "The learner remains confused after the standard aligned-equation construction."),
    c("E14", 4, "A population becomes 1,331 after growing by 10% each year for 3 years. Find the starting population.",
      [("teacher", "What three-year growth factor produced the final population?"),
       ("student", "I don't know how to work backward through percentage growth."),
       ("teacher", "The total multiplier is 1.1^3."),
       ("student", "I can calculate the multiplier, but I don't know what to do with 1,331."),
       ("teacher", "Divide the final population by the total multiplier to reverse the growth."),
       ("student", "I still cannot set up the starting-value calculation.")],
      "The learner remains confused after the reverse-growth operation is specified."),
    c("E15", 4, "Vectors a = (2, -3) and b = (-1, 5). Find 3a - 2b.",
      [("teacher", "How would you begin with the scalar multiples?"),
       ("student", "I don't know how to multiply a vector by a number."),
       ("teacher", "Multiply every component of a by 3 and every component of b by 2."),
       ("student", "I found 3a, but I don't know how to subtract 2b."),
       ("teacher", "Subtract corresponding first components and corresponding second components."),
       ("student", "I am still unable to combine the two vectors.")],
      "Repeated confusion remains after both scalar and component operations are given."),
    c("E16", 4, "Rewrite x^2 - 8x + 3 in completed-square form.",
      [("teacher", "What number belongs inside the squared bracket?"),
       ("student", "I don't know how to choose it."),
       ("teacher", "Take half of -8, giving -4, so begin with (x - 4)^2."),
       ("student", "I have the bracket but don't know how to keep the expression equal."),
       ("teacher", "Expanding the bracket creates +16, so compensate outside the bracket."),
       ("student", "I still cannot decide what constant belongs outside.")],
      "The learner remains confused after the bracket and compensation idea are supplied."),

    c("F01", 5, "Rationalise the denominator of 5/sqrt(3).",
      [("teacher", "What could you multiply by without changing the value?"),
       ("student", "I tried multiplying only the denominator by sqrt(3)."),
       ("teacher", "Multiply numerator and denominator by the same sqrt(3)."),
       ("student", "I can write the multiplication, but I do not see why the denominator becomes 3."),
       ("teacher", "Use sqrt(3) times sqrt(3) = 3."),
       ("student", "Please explain the entire rationalising method directly and show the simplified result.")],
      "After multiple supports, the learner explicitly requests a direct worked explanation."),
    c("F02", 5, "Rewrite x^2 + 10x - 6 in completed-square form.",
      [("teacher", "What half-coefficient would appear in the bracket?"),
       ("student", "I used 10 inside the bracket."),
       ("teacher", "Use half of 10, so the bracket begins (x + 5)^2."),
       ("student", "I know the bracket but not the outside correction."),
       ("teacher", "Compare the +25 produced by the bracket with the original constant -6."),
       ("student", "Could you now show the completed-square conversion step by step and explain the correction?")],
      "The learner explicitly requests a full explanation after two targeted scaffolds."),
    c("F03", 5, "Calculate (6 × 10^8) / (3 × 10^-2) in scientific notation.",
      [("teacher", "How do the coefficients combine?"),
       ("student", "Six divided by three gives 2, but the powers confuse me."),
       ("teacher", "For division, subtract the denominator exponent from the numerator exponent."),
       ("student", "I see 8 - (-2), but I am uncertain why that becomes addition."),
       ("teacher", "Subtracting a negative exponent adds its magnitude."),
       ("student", "Please work through the exponent rule directly and give the final standard-form value.")],
      "The learner asks for direct explanation after the calculation has been narrowed."),
    c("F04", 5, "A box contains 3 green and 5 yellow balls. Two are drawn without replacement. Find the probability of one of each colour.",
      [("teacher", "Which two colour orders are possible?"),
       ("student", "I only wrote green then yellow."),
       ("teacher", "Also include yellow then green."),
       ("student", "I can name both paths, but I am not sure how their changing denominators work."),
       ("teacher", "Each second draw has 7 balls remaining, and the two mutually exclusive path probabilities are added."),
       ("student", "Please show both path calculations and explain why they are added.")],
      "The learner directly requests the full method after path-level supports."),
    c("F05", 5, "Solve 2x + 3y = 17 and 4x - 3y = 7.",
      [("teacher", "How could the y terms be eliminated?"),
       ("student", "I tried subtracting the equations and got more y terms."),
       ("teacher", "Add the equations because +3y and -3y cancel."),
       ("student", "Adding gives 6x = 24, but I am unsure how to finish both variables."),
       ("teacher", "Find x from 6x = 24, then substitute it into either original equation."),
       ("student", "Can you explain the elimination method directly and work through the substitution too?")],
      "After prior support, the learner explicitly asks for a complete worked method."),
    c("F06", 5, "A triangle has sides 9 cm and 13 cm with included angle 47 degrees. Find the third side.",
      [("teacher", "Which rule matches the given information?"),
       ("student", "I chose Pythagoras, but the triangle is not right-angled."),
       ("teacher", "Use the cosine rule because two sides and the included angle are known."),
       ("student", "I found the formula but am uncertain about the negative cosine term."),
       ("teacher", "Substitute the known sides around -2ab cos C before taking a square root."),
       ("student", "Please demonstrate the cosine-rule substitution and calculation step by step.")],
      "The learner explicitly requests a worked explanation after two supports."),
    c("F07", 5, "A histogram class interval 12 to 20 has frequency 36. Find its frequency density.",
      [("teacher", "What class width did you find?"),
       ("student", "I used 20 as the width."),
       ("teacher", "Class width is upper boundary minus lower boundary."),
       ("student", "The width is 8, but I do not understand why frequency is divided by it."),
       ("teacher", "Density represents frequency per unit of class width."),
       ("student", "Could you explain the density idea directly and show the calculation?")],
      "The learner requests direct explanation after procedural and conceptual scaffolds."),
    c("F08", 5, "Solve x^2 - 6x - 7 = 0 by completing the square.",
      [("teacher", "How could you form a square from x^2 - 6x?"),
       ("student", "I wrote (x - 6)^2."),
       ("teacher", "Use half of -6 inside the bracket."),
       ("student", "That gives (x - 3)^2, but I do not know how the equation changes."),
       ("teacher", "Add and subtract 9, then isolate the squared bracket."),
       ("student", "Please show the complete solution by completing the square and explain each equality.")],
      "The learner explicitly requests the whole method after two targeted cues."),
    c("F09", 5, "Calculate 7/9 divided by 14/15.",
      [("teacher", "What operation did you use between the fractions?"),
       ("student", "I divided numerator by numerator and denominator by denominator."),
       ("teacher", "Rewrite division by a fraction as multiplication by its reciprocal."),
       ("student", "I flipped 14/15, but I don't understand why that is valid."),
       ("teacher", "A reciprocal is the number that makes a product of one with the divisor."),
       ("student", "Please explain the reciprocal rule directly and work through the simplification.")],
      "The learner asks for a direct conceptual and procedural explanation."),
    c("F10", 5, "Rearrange A = (h/2)(a + b) to make h the subject.",
      [("teacher", "How could you remove the division by 2?"),
       ("student", "I multiplied A by 2, but I am not sure what happens to the bracket."),
       ("teacher", "After multiplying by 2, write 2A = h(a + b)."),
       ("student", "I know h is multiplied by the bracket but do not know the final inverse step."),
       ("teacher", "Divide both sides by the entire factor a + b."),
       ("student", "Can you show the rearrangement from start to finish and explain why the bracket stays together?")],
      "The learner explicitly requests full explanation after two inverse-operation supports."),
    c("F11", 5, "Find the nth term of 11, 18, 25, 32, ...",
      [("teacher", "What is the common difference?"),
       ("student", "The difference is 7, so I wrote 7n."),
       ("teacher", "Compare 7n's first term with the sequence's first term."),
       ("student", "7n starts at 7, so I need an adjustment of 4, but I do not see why."),
       ("teacher", "The constant adjustment aligns every term after the common difference is fixed."),
       ("student", "Please explain the nth-term construction directly and verify it on several terms.")],
      "The learner requests direct explanation after identifying both rule components."),
    c("F12", 5, "After a 20% discount, a bicycle costs $360. Find its original price.",
      [("teacher", "What percentage of the original price remains?"),
       ("student", "Eighty percent remains, but I subtracted 20 from 360."),
       ("teacher", "Represent the sale price as 0.8 times the original price."),
       ("student", "I can write 0.8p = 360 but do not understand how to reverse it."),
       ("teacher", "Undo multiplication by 0.8 using division."),
       ("student", "Please show the reverse-percentage calculation and explain why division recovers the original.")],
      "The learner explicitly requests a worked reverse-percentage explanation."),
    c("F13", 5, "A sector has radius 12 cm and angle 75 degrees. Find its area in terms of pi.",
      [("teacher", "What fraction of the circle is the sector?"),
       ("student", "I used 75/100 because 75 looks like a percentage."),
       ("teacher", "Compare the angle with a full 360-degree turn."),
       ("student", "The fraction is 75/360, but I am unsure how it combines with the circle area."),
       ("teacher", "Multiply that fraction by pi times radius squared."),
       ("student", "Could you explain the sector-area method directly and simplify the result?")],
      "The learner asks for direct explanation after both fraction and area cues."),
    c("F14", 5, "Evaluate 5^-2.",
      [("teacher", "What did you think the negative exponent meant?"),
       ("student", "I wrote -25."),
       ("teacher", "A negative exponent indicates the reciprocal of the positive power."),
       ("student", "I can write 1 over something, but I do not understand why the exponent causes that."),
       ("teacher", "Use the exponent law 5^2 times 5^-2 = 5^0 = 1."),
       ("student", "Please explain the negative-exponent rule directly and give the final value.")],
      "The learner explicitly asks for conceptual explanation after two supports."),
    c("F15", 5, "A box plot has lower quartile 22 and upper quartile 47. Find and interpret the interquartile range.",
      [("teacher", "What calculation gives the interquartile range?"),
       ("student", "I added 22 and 47."),
       ("teacher", "The interquartile range measures the span from Q1 to Q3."),
       ("student", "I think I subtract, but I don't know what the result means."),
       ("teacher", "Compute Q3 - Q1; it describes the spread of the middle half of the data."),
       ("student", "Please explain the calculation and interpretation together using these values.")],
      "The learner asks for a direct explanation after calculation and meaning have been scaffolded."),
    c("F16", 5, "On a map, 4 cm represents 18 km. A route measures 11 cm. Find the actual distance.",
      [("teacher", "How could you find the distance represented by one centimetre?"),
       ("student", "I multiplied 18 by 4."),
       ("teacher", "Divide 18 by 4 to obtain kilometres per centimetre."),
       ("student", "I get 4.5 km per centimetre, but I am unsure how to use the 11 cm route."),
       ("teacher", "Scale the unit distance by 11."),
       ("student", "Could you show the complete scale calculation and explain each multiplier?")],
      "After two scaffolds, the learner directly requests the complete method."),

    c("G01", 6, "Solve 7x + 5 = 47.",
      [("teacher", "What did you do first?"),
       ("student", "I divided 47 by 7 before removing 5."),
       ("teacher", "Undo the addition of 5 first, then divide by 7."),
       ("student", "Subtracting 5 gives 7x = 42, and dividing by 7 gives x = 6.")],
      "The learner fully corrects the inverse-operation order and solution."),
    c("G02", 6, "Calculate 3/4 - 2/9.",
      [("teacher", "How did you first subtract the fractions?"),
       ("student", "I subtracted top and bottom to get 1/-5."),
       ("teacher", "Use a common denominator of 36 before subtracting numerators."),
       ("student", "Three quarters is 27/36 and two ninths is 8/36, so the result is 19/36.")],
      "The learner uses the scaffold to complete the fraction subtraction correctly."),
    c("G03", 6, "A shop increases a $250 price by 12%. Find the new price.",
      [("teacher", "What did you initially add?"),
       ("student", "I added 12 dollars and got $262."),
       ("teacher", "Calculate 12 percent of 250 before adding the increase."),
       ("student", "The increase is 0.12 times 250 = 30, so the new price is $280.")],
      "The learner recovers and completes a correct percentage increase."),
    c("G04", 6, "Find the area of a trapezium with parallel sides 8 cm and 14 cm and height 5 cm.",
      [("teacher", "What formula did you first use?"),
       ("student", "I multiplied 8 by 14."),
       ("teacher", "Average the parallel sides, then multiply by the perpendicular height."),
       ("student", "The average is 11, and 11 times 5 gives an area of 55 square centimetres.")],
      "The learner correctly applies the scaffold and completes the area."),
    c("G05", 6, "Expand and simplify 4(3x - 2) + x.",
      [("teacher", "What happened in your first expansion?"),
       ("student", "I wrote 12x - 2 + x."),
       ("teacher", "Distribute 4 to both terms before collecting like terms."),
       ("student", "That gives 12x - 8 + x, so the simplified expression is 13x - 8.")],
      "The learner corrects distribution and combines terms accurately."),
    c("G06", 6, "Find the probability of rolling an even number on a fair six-sided die.",
      [("teacher", "Which outcomes did you first count?"),
       ("student", "I counted 2 and 4 but forgot 6, so I wrote 2/6."),
       ("teacher", "List every even face from 1 through 6."),
       ("student", "The even faces are 2, 4, and 6, so the probability is 3/6 = 1/2.")],
      "The learner responds to one cue with a complete correct outcome count."),
    c("G07", 6, "Translate the point (5, -3) by the vector (-2, 7).",
      [("teacher", "How did you first use the vector?"),
       ("student", "I changed both coordinate signs."),
       ("teacher", "Add each vector component to its matching coordinate."),
       ("student", "The image is (5 - 2, -3 + 7) = (3, 4).")],
      "The learner correctly applies component addition and completes the translation."),
    c("G08", 6, "Simplify (a^3)^4.",
      [("teacher", "What exponent did you initially use?"),
       ("student", "I added 3 and 4 and got a^7."),
       ("teacher", "A power raised to another power multiplies the exponents."),
       ("student", "Then the exponent is 3 times 4 = 12, so the result is a^12.")],
      "The learner corrects the exponent law and reaches the final answer."),
    c("G09", 6, "Find the gradient of y = -2x + 9.",
      [("teacher", "Which number did you first call the gradient?"),
       ("student", "I used 9 because it is the positive number."),
       ("teacher", "In y = mx + c, the coefficient m multiplying x is the gradient."),
       ("student", "The coefficient of x is -2, so the gradient is -2 and 9 is the intercept.")],
      "The learner correctly distinguishes gradient and intercept after one scaffold."),
    c("G10", 6, "A right triangle has legs 9 cm and 12 cm. Find its hypotenuse.",
      [("teacher", "What did you do with the two legs at first?"),
       ("student", "I added 9 and 12 and got 21."),
       ("teacher", "Use the sum of the squares of the legs, then take the square root."),
       ("student", "9^2 + 12^2 = 81 + 144 = 225, and the square root is 15 cm.")],
      "The learner follows the scaffold through a complete correct calculation."),
    c("G11", 6, "Factor x^2 + 13x + 36.",
      [("teacher", "Which pair did you first choose?"),
       ("student", "I chose 6 and 6 because they multiply to 36."),
       ("teacher", "The pair must also add to 13."),
       ("student", "Four and nine multiply to 36 and add to 13, so the factors are (x + 4)(x + 9).")],
      "The learner uses the second condition and completes the factorization."),
    c("G12", 6, "Convert 2.75 hours to hours and minutes.",
      [("teacher", "How did you first interpret 0.75 of an hour?"),
       ("student", "I called it 75 minutes."),
       ("teacher", "Multiply the fractional hour by 60 minutes."),
       ("student", "0.75 times 60 is 45, so the time is 2 hours 45 minutes.")],
      "The learner corrects the unit interpretation and reaches the final conversion."),
    c("G13", 6, "Find the midpoint between (2, -5) and (8, 7).",
      [("teacher", "What did you initially do with the coordinates?"),
       ("student", "I subtracted the points and got (6, 12)."),
       ("teacher", "Average the x-coordinates and average the y-coordinates separately."),
       ("student", "The averages are 5 and 1, so the midpoint is (5, 1).")],
      "The learner fully repairs the coordinate method and answer."),
    c("G14", 6, "A sequence begins 30, 24, 18, 12, ... Find the next term and describe the rule.",
      [("teacher", "What pattern did you first propose?"),
       ("student", "I thought the terms were halving."),
       ("teacher", "Compare consecutive differences instead."),
       ("student", "Each difference is -6, so the next term is 6 and the rule is subtract 6.")],
      "The learner correctly recovers both next term and sequence rule."),
    c("G15", 6, "A circle has radius 7 cm. Find its diameter and circumference in terms of pi.",
      [("teacher", "What did you first use as the diameter?"),
       ("student", "I said the diameter was 7 cm."),
       ("teacher", "The diameter spans two radii."),
       ("student", "The diameter is 14 cm, so the circumference is pi times 14, or 14pi cm.")],
      "The learner correctly applies the radius-diameter relationship and finishes."),
    c("G16", 6, "Solve the inequality 4x + 1 > 17.",
      [("teacher", "What did you initially do?"),
       ("student", "I divided 17 by 4 before removing 1."),
       ("teacher", "Subtract 1 from both sides before dividing by positive 4."),
       ("student", "That gives 4x > 16, then x > 4; the direction stays the same because 4 is positive.")],
      "The learner recovers the operation order and explicitly checks the inequality direction."),

    c("H01", 7, "Solve 8x - 11 = 53.",
      [("teacher", "What did you do first?"),
       ("student", "I divided 53 by 8 and then subtracted 11."),
       ("teacher", "Undo minus 11 before dividing by 8."),
       ("student", "Adding 11 gives 8x = 64. I know division is next, but I have not written x yet.")],
      "The learner recovers the correct intermediate equation and next operation."),
    c("H02", 7, "Calculate 7/10 + 5/12.",
      [("teacher", "How did you first combine the fractions?"),
       ("student", "I added across and got 12/22."),
       ("teacher", "Use a common denominator divisible by both 10 and 12."),
       ("student", "A common denominator is 60, so the fractions become 42/60 and 25/60; I still need to add and simplify.")],
      "The learner has correct equivalent fractions but has not completed the sum."),
    c("H03", 7, "A $480 television is discounted by 15%. Find the sale price.",
      [("teacher", "What did you initially subtract?"),
       ("student", "I subtracted 15 dollars."),
       ("teacher", "Find 15 percent of 480 as the discount amount."),
       ("student", "The discount is 0.15 times 480 = 72, so I should subtract 72 from 480 next.")],
      "The learner finds the correct discount and identifies but has not executed the final step."),
    c("H04", 7, "Find the area of a circle with diameter 16 cm in terms of pi.",
      [("teacher", "What value did you first use as the radius?"),
       ("student", "I used 16 as the radius."),
       ("teacher", "The radius is half the diameter before applying A = pi r^2."),
       ("student", "The radius is 8, so the area setup is pi times 8 squared; I have not simplified it yet.")],
      "The learner corrects the radius and forms the right expression but stops before simplification."),
    c("H05", 7, "Expand (x - 3)(x + 7).",
      [("teacher", "What did your first expansion include?"),
       ("student", "I only multiplied x by x and -3 by 7."),
       ("teacher", "Include the two cross-products as well."),
       ("student", "I now have x^2 + 7x - 3x - 21. I think the middle terms combine, but I have not done it.")],
      "The learner repairs the expansion structure and needs only local collection."),
    c("H06", 7, "A fair spinner numbered 1 to 8 is spun once. Find the probability of a prime number.",
      [("teacher", "Which numbers did you first call prime?"),
       ("student", "I included 1 and missed 2."),
       ("teacher", "One is not prime, and two is the first prime."),
       ("student", "The prime outcomes are 2, 3, 5, and 7, so I have four favorable outcomes out of eight; I still need to simplify.")],
      "The learner corrects the outcome set and forms the probability but has not simplified it."),
    c("H07", 7, "Reflect the point (6, -4) across the y-axis.",
      [("teacher", "What did you first change?"),
       ("student", "I changed both coordinate signs."),
       ("teacher", "A y-axis reflection changes only the x-coordinate sign."),
       ("student", "Then x becomes -6 while y stays -4. I think the image is (-6, -4), but I want to verify.")],
      "The learner correctly applies the rule but presents the result tentatively."),
    c("H08", 7, "Simplify r^11 / r^6 for nonzero r.",
      [("teacher", "What did you first do with the exponents?"),
       ("student", "I divided 11 by 6."),
       ("teacher", "A quotient of equal bases subtracts exponents."),
       ("student", "The new exponent is 11 - 6 = 5, so I think the expression is r^5, though I have not checked by cancellation.")],
      "The learner reaches the correct rule and tentative answer after one scaffold."),
    c("H09", 7, "Find the equation of a line with gradient 3 and y-intercept -5.",
      [("teacher", "What equation did you first write?"),
       ("student", "I wrote y = -5x + 3."),
       ("teacher", "In y = mx + c, m is the gradient and c is the y-intercept."),
       ("student", "Then m = 3 and c = -5, so the equation starts y = 3x - 5; I think that is complete.")],
      "The learner correctly swaps the roles and offers a tentative complete equation."),
    c("H10", 7, "A right triangle has hypotenuse 20 cm and one leg 12 cm. Find the other leg.",
      [("teacher", "How did you first arrange the theorem?"),
       ("student", "I added 20 squared and 12 squared."),
       ("teacher", "The unknown leg squared equals hypotenuse squared minus the known leg squared."),
       ("student", "So the unknown squared is 20^2 - 12^2 = 400 - 144 = 256. I still need to take the square root.")],
      "The learner has the correct squared value but has not completed the root."),
    c("H11", 7, "Factor x^2 - x - 12.",
      [("teacher", "Which signs did you first choose?"),
       ("student", "I chose 3 and 4, both positive."),
       ("teacher", "The product is negative, so the factor numbers need opposite signs and must sum to -1."),
       ("student", "I think -4 and 3 work because their product is -12 and sum is -1; I have not written the brackets.")],
      "The learner identifies the correct signed pair but has not completed factor notation."),
    c("H12", 7, "Convert 4.2 metres to centimetres.",
      [("teacher", "What did you first multiply by?"),
       ("student", "I multiplied by 10 and got 42."),
       ("teacher", "One metre contains 100 centimetres."),
       ("student", "Then I should multiply 4.2 by 100. That seems to give 420, but I want to check the unit.")],
      "The learner recovers the correct factor and tentative value but checks the unit."),
    c("H13", 7, "Find the midpoint of (-3, 8) and (9, 2).",
      [("teacher", "How did you first combine the points?"),
       ("student", "I added coordinates without dividing and got (6, 10)."),
       ("teacher", "After adding corresponding coordinates, divide each total by 2."),
       ("student", "The midpoint coordinates are 6/2 and 10/2, so I think the point is (3, 5).")],
      "The learner completes the corrected averaging with tentative confidence."),
    c("H14", 7, "Find the nth term of 5, 13, 21, 29, ...",
      [("teacher", "What rule did you first try?"),
       ("student", "I wrote 8n because the difference is 8."),
       ("teacher", "Compare the first value of 8n with the actual first term."),
       ("student", "At n = 1, 8n gives 8, which is 3 too high, so I think the rule is 8n - 3; I have not checked later terms.")],
      "The learner derives a plausible correct rule but has not verified it."),
    c("H15", 7, "A cylinder has radius 4 cm and height 9 cm. Find its volume in terms of pi.",
      [("teacher", "What did you first square?"),
       ("student", "I squared the height."),
       ("teacher", "Cylinder volume uses the circular base area pi r^2 times height."),
       ("student", "The setup is pi times 4 squared times 9. I get 16 times 9 pi, but I have not multiplied it.")],
      "The learner corrects the formula and reaches the last arithmetic step."),
    c("H16", 7, "Solve 2x + 3 > 15.",
      [("teacher", "What did you first do?"),
       ("student", "I divided 15 by 2 before subtracting 3."),
       ("teacher", "Subtract 3 from both sides before dividing by positive 2."),
       ("student", "That gives 2x > 12 and then x > 6. I think the sign stays the same because I divided by a positive number.")],
      "The learner recovers the solution and explicitly reasons about the sign, but remains tentative."),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_text(value: object) -> str:
    text = str(value).casefold().replace("tutor:", "teacher:")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def history_text(history: object) -> str:
    if isinstance(history, str):
        return history
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        lines: list[str] = []
        for turn in history:
            if isinstance(turn, dict):
                user = turn.get("user", turn.get("role", ""))
                text = turn.get("text", turn.get("content", ""))
                lines.append(f"{user}: {text}")
        return "\n".join(lines)
    return str(history)


def fingerprint(problem: object, history: object) -> str:
    return normalize_text(problem) + " || " + normalize_text(history_text(history))


def excluded_records() -> dict[str, list[dict[str, Any]]]:
    sources: dict[str, list[dict[str, Any]]] = {}
    sources["review1_frozen_48"] = [
        {"problem": row["problem"], "history": row["dialogue_history"]}
        for row in read_jsonl(REVIEW1 / "fresh_review_states.jsonl")
    ]

    with (OLD_DIAGNOSTIC / "telling_boundary_challenge.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    sources["old_telling_boundary_48"] = [
        {"problem": row["problem"], "history": json.loads(row["history"])}
        for row in rows
    ]

    with (OLD_DIAGNOSTIC / "real_state_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    sources["collected_real_states_87"] = [
        {
            "problem": row["problem"],
            "history": json.loads(row["history_before_action"]),
        }
        for row in rows
    ]

    for source, filename in (
        ("c1_correction_train", "correction_train.jsonl"),
        ("prior_synthetic_calibration_train", "telling_calibration_train_v1.jsonl"),
        ("prior_synthetic_calibration_validation", "telling_calibration_validation_v1.jsonl"),
        ("prior_synthetic_boundary_challenge", "telling_boundary_challenge_val_v1.jsonl"),
    ):
        sources[source] = [
            {
                "problem": row.get("problem", ""),
                "history": row.get("history", row.get("dialogue_history", "")),
            }
            for row in read_jsonl(PREPARATION / filename)
        ]
    return sources


def audit_and_freeze() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(DRAFT_CASES) != 128:
        raise AssertionError(f"Expected 128 authored states, found {len(DRAFT_CASES)}")
    expected_balance = Counter({stratum: 16 for stratum in STRATA})
    observed_balance = Counter(row["semantic_stratum"] for row in DRAFT_CASES)
    if observed_balance != expected_balance:
        raise AssertionError(f"Expected 16 states per stratum, got {observed_balance}")
    if len({row["draft_id"] for row in DRAFT_CASES}) != 128:
        raise AssertionError("Draft IDs are not unique")

    draft_fingerprints = [
        fingerprint(row["problem"], row["dialogue_history"])
        for row in DRAFT_CASES
    ]
    if len(set(draft_fingerprints)) != 128:
        raise AssertionError("The authored pool contains duplicate states")

    prior = excluded_records()
    prior_texts = {
        source: [fingerprint(row["problem"], row["history"]) for row in rows]
        for source, rows in prior.items()
    }
    prior_profiles = {
        source: [(known, len(known), Counter(known)) for known in texts]
        for source, texts in prior_texts.items()
    }
    exact_overlap = {
        source: sum(current in set(texts) for current in draft_fingerprints)
        for source, texts in prior_texts.items()
    }
    if any(exact_overlap.values()):
        raise AssertionError(f"Excluded-source exact overlap detected: {exact_overlap}")

    # Apply the same SequenceMatcher rejection logic prospectively across all
    # excluded review, diagnostic, correction, and synthetic boundary sources.
    nearest: list[dict[str, Any]] = []
    global_max = -1.0
    global_source = ""
    global_draft = ""
    for row, current in zip(DRAFT_CASES, draft_fingerprints, strict=True):
        current_length = len(current)
        current_counts = Counter(current)
        best_ratio = -1.0
        best_source = ""
        best_index = -1
        for source, profiles in prior_profiles.items():
            for index, (known, known_length, known_counts) in enumerate(profiles):
                denominator = current_length + known_length
                # These are the exact formulas used by SequenceMatcher's two
                # quick upper bounds. Computing them from cached profiles avoids
                # constructing a matcher for comparisons that cannot win.
                real_quick_ratio = 2.0 * min(current_length, known_length) / denominator
                if real_quick_ratio <= best_ratio:
                    continue
                matches = sum((current_counts & known_counts).values())
                quick_ratio = 2.0 * matches / denominator
                if quick_ratio <= best_ratio:
                    continue
                ratio = SequenceMatcher(None, current, known).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_source = source
                    best_index = index
        nearest.append(
            {
                "draft_id": row["draft_id"],
                "excluded_source": best_source,
                "excluded_source_index": best_index,
                "sequence_similarity": float(best_ratio),
            }
        )
        if best_ratio > global_max:
            global_max = best_ratio
            global_source = best_source
            global_draft = row["draft_id"]
    if global_max >= SIMILARITY_THRESHOLD:
        raise AssertionError(
            "Fresh-state similarity threshold breached before inference: "
            f"draft={global_draft} source={global_source} similarity={global_max}"
        )

    shuffled = list(DRAFT_CASES)
    random.Random(REVIEW_SET_SEED).shuffle(shuffled)
    frozen: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for row in shuffled:
        opaque = hashlib.sha256(
            f"review2:{REVIEW_SET_SEED}:{row['draft_id']}".encode("utf-8")
        ).hexdigest()[:10].upper()
        case_id = f"R2-{opaque}"
        if case_id in used_ids:
            raise AssertionError("Opaque Review 2 case-ID collision")
        used_ids.add(case_id)
        frozen.append(
            {
                "case_id": case_id,
                "semantic_stratum": row["semantic_stratum"],
                "problem": row["problem"],
                "dialogue_history": row["dialogue_history"],
                "latest_student_state": row["latest_student_state"],
            }
        )
    return frozen, {
        "authorship": "fresh_independently_authored_deployment_style_synthetic_v2",
        "exact_overlap_counts": exact_overlap,
        "similarity_method": "difflib.SequenceMatcher on normalized problem plus dialogue",
        "similarity_rejection_threshold": SIMILARITY_THRESHOLD,
        "maximum_similarity_to_any_excluded_prior_material": float(global_max),
        "maximum_similarity_draft_id": global_draft,
        "maximum_similarity_source": global_source,
        "nearest_excluded_state_by_draft": nearest,
        "review1_labels_read_or_used": False,
        "outcome_signals_read_or_used": False,
        "mathdial_final_test_read": False,
        "mrbench_v3_test_read": False,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def verify_model(model_dir: Path, expected_weight: str) -> dict[str, str]:
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete model export {model_dir}: missing={missing}")
    hashes = {name: sha256(model_dir / name) for name in required}
    if hashes["model.safetensors"] != expected_weight:
        raise ValueError(f"Weight hash mismatch for {model_dir}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    observed = {int(index): label for index, label in config["id2label"].items()}
    if observed != dict(enumerate(MOVES)):
        raise ValueError(f"Move order mismatch for {model_dir}: {observed}")
    return hashes


def load_frozen_inference():
    analysis_path = OLD_DIAGNOSTIC / "analysis.py"
    if sha256(analysis_path) != EXPECTED_V1_ANALYSIS_HASH:
        raise ValueError("Frozen inference implementation hash mismatch")
    spec = importlib.util.spec_from_file_location("review2_frozen_inference", analysis_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen inference implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vector_fields(prefix: str, vector: np.ndarray) -> dict[str, Any]:
    values = [float(value) for value in vector.tolist()]
    return {
        **{f"{prefix}_p_{move}": values[index] for index, move in enumerate(MOVES)},
        f"{prefix}_top1": MOVES[int(np.argmax(vector))],
    }


def reviewer_instructions(states: list[dict[str, Any]], state_hash: str) -> str:
    blocks: list[str] = []
    for index, state in enumerate(states, start=1):
        history = "\n".join(
            f"{turn['user']}: {turn['text']}" for turn in state["dialogue_history"]
        )
        blocks.append(
            f"""### Case {index}: {state['case_id']}

**Problem**

{state['problem']}

**Conversation/history**

```text
{history}
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
"""
        )
    return f"""# Blinded Pedagogical-Move Expert Review — Extension 2

## Reviewer protocol

Review every case independently using only the problem and conversation shown. Choose exactly one most appropriate next pedagogical move. Do not consult Review 1 or any learner outcome, model output, or external learner record.

Use these fixed definitions for every case:

- **generic:** neutral encouragement, transition, or broadly supportive response without targeting a specific reasoning gap.
- **probing:** ask the learner to explain, reason, recall, or reveal their thinking.
- **focus:** direct attention to a specific error, relationship, clue, or sub-step without directly giving the explanation or solution.
- **telling:** provide explicit instruction, explanation, method, or missing step when further equivalent probing or focusing is unlikely to help.

Confidence:

- `1` = low
- `2` = medium
- `3` = high

Enter judgments only in `expert_review_form_v2.csv`. Leave `expert_reason` empty or use one sentence maximum. Do not alter case IDs or add rows.

Frozen extension state-set SHA-256: `{state_hash}`. This packet contains {len(states)} new model-disagreement cases.

## Review cases

{''.join(blocks)}"""


def preregistration_text(
    state_hash: str,
    counts: dict[str, int],
    audit: dict[str, Any],
    baseline_hashes: dict[str, str],
    candidate_hashes: dict[str, str],
) -> str:
    count_lines = "\n".join(f"- `{name}`: {counts[name]}" for name in STRATA)
    return f"""# Preregistration: MD7-R2-TELL-C1 Blinded Review Extension 2

This study is independent of Review 1. Review 1 remains D (inconclusive), with 3 candidate wins, 0 baseline wins, 2 neither, and 3 decisive disagreements. Its expert labels are not used to author, select, or judge Review 2 states.

## Frozen models

- Reference weight SHA-256: `{baseline_hashes['model.safetensors']}`
- Recalibrated weight SHA-256: `{candidate_hashes['model.safetensors']}`
- Move order: `generic`, `probing`, `focus`, `telling`

## Prospectively frozen pool

- Rows: 128
- SHA-256: `{state_hash}`
- Maximum excluded-material similarity: `{audit['maximum_similarity_to_any_excluded_prior_material']:.12f}`
- Rejection rule: reject any state with normalized SequenceMatcher similarity >= `{SIMILARITY_THRESHOLD:.2f}` before inference.

{count_lines}

The entire pool is authored, audited, written, and hashed before either model is loaded. No state may be edited after model predictions are observed.

## Packet rule

All and only new reference-versus-recalibrated top-1 disagreements enter the primary packet. No disagreement may be cherry-picked. Agreement padding is not used. If fewer than {MIN_MODEL_DISAGREEMENTS} model disagreements occur, preparation stops with insufficient disagreement support and no model conclusion.

## Blinding

Reviewers see only opaque case IDs, problems, dialogue histories, fixed move definitions, confidence 1-3, and an optional one-sentence reason. They do not see model identities, System X/Y, probabilities, strata, BKT, MRB1, evaluator outcomes, historical outcomes, or Review 1 labels. A deterministic seed `{MODEL_BLINDING_SEED}` seals model identity as System X/Y in a separate file.

## Primary endpoint and stopping rule

Primary analysis is restricted to new model-disagreement cases. Each is classified as candidate win, baseline win, or neither. Decisive count is candidate wins plus baseline wins. If decisive count is below {MIN_DECISIVE_DISAGREEMENTS}, Review 2 is D: REVIEW STILL INCONCLUSIVE. If it is at least {MIN_DECISIVE_DISAGREEMENTS}, raw wins and an exact two-sided binomial test are reported.

## Secondary endpoints

Within the reviewed extension packet: exact agreement, Macro-F1, per-class F1, telling precision/recall, false telling among expert non-telling cases, and telling counts in C, G/H, and D/E/F. Every C/G/H disagreement is printed after review with problem, history, expert move/confidence, and both model moves.

## Operational decision rule

1. D if decisive new disagreements < {MIN_DECISIVE_DISAGREEMENTS} or an integrity check fails.
2. B if, among expert-non-telling C/G/H cases, the recalibrated model has at least 2 more false-telling predictions than the reference or a false-telling-rate increase of at least 0.20.
3. A only if decisive count >= {MIN_DECISIVE_DISAGREEMENTS}, recalibrated wins exceed reference wins, recalibrated telling recall in expert-telling D/E/F cases is at least reference recall, and rule 2 is false.
4. Otherwise C.

The formal decision depends on Review 2 alone. Only after that decision is frozen, an explicitly exploratory cumulative table adds Review 1's 3 candidate wins and 0 baseline wins. No outcome authorizes promotion.

## Safety

Training = 0; parameter changes = 0; production changes = 0; Tutor API calls = 0; BKT/LinTS updates = 0; authoritative-data writes = 0; MathDial final-test use = 0; MRBench V3 test use = 0; promotion = 0.
"""


def main() -> None:
    final_artifacts = (
        "fresh_review_states_v2.jsonl",
        "fresh_review_states_v2_manifest.json",
        "hidden_model_predictions_v2.csv",
        "hidden_model_mapping_v2.json",
        "expert_review_form_v2.csv",
        "expert_review_instructions_v2.md",
        "preregistration.md",
    )
    if all((OUT / name).is_file() for name in final_artifacts):
        manifest = json.loads(
            (OUT / "fresh_review_states_v2_manifest.json").read_text(encoding="utf-8")
        )
        if sha256(OUT / "fresh_review_states_v2.jsonl") != manifest["review_set"]["sha256"]:
            raise RuntimeError("Existing frozen Review 2 state hash is invalid")
        print(json.dumps({
            "status": "ALREADY_PREPARED_NO_FILES_CHANGED",
            "review_set_sha256": manifest["review_set"]["sha256"],
            "disagreements": manifest["comparison"]["disagreement_count"],
            "expert_packet": manifest["comparison"]["packet_count"],
        }, indent=2))
        return

    protected = {
        "frozen_inference": OLD_DIAGNOSTIC / "analysis.py",
        "review1_states": REVIEW1 / "fresh_review_states.jsonl",
        "review1_analysis": REVIEW1 / "review_analysis.json",
        "old_boundary_48": OLD_DIAGNOSTIC / "telling_boundary_challenge.csv",
        "real_states_87": OLD_DIAGNOSTIC / "real_state_predictions.csv",
        "correction_train": PREPARATION / "correction_train.jsonl",
        "calibration_train": PREPARATION / "telling_calibration_train_v1.jsonl",
        "calibration_validation": PREPARATION / "telling_calibration_validation_v1.jsonl",
        "calibration_challenge": PREPARATION / "telling_boundary_challenge_val_v1.jsonl",
        "baseline_weight": BASELINE / "model.safetensors",
        "candidate_weight": CANDIDATE / "model.safetensors",
    }
    before_hashes = {name: sha256(path) for name, path in protected.items()}
    baseline_hashes = verify_model(BASELINE, EXPECTED_BASELINE_WEIGHT)
    candidate_hashes = verify_model(CANDIDATE, EXPECTED_CANDIDATE_WEIGHT)

    # Phase 1: audit, freeze, hash, manifest, and preregister. No model module
    # has been imported and no prediction exists before this phase completes.
    frozen, audit = audit_and_freeze()
    states_path = OUT / "fresh_review_states_v2.jsonl"
    write_jsonl(states_path, frozen)
    state_hash = sha256(states_path)
    counts = dict(Counter(row["semantic_stratum"] for row in frozen))
    frozen_at = datetime.now(UTC).isoformat()
    prereg = preregistration_text(
        state_hash, counts, audit, baseline_hashes, candidate_hashes
    )
    (OUT / "preregistration.md").write_text(prereg, encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": "md7_r2_tell_c1_blinded_review_manifest_v2",
        "status": "frozen_before_model_inference",
        "review_set": {
            "path": "fresh_review_states_v2.jsonl",
            "classification": "INTERNAL_NOT_FOR_REVIEWER",
            "row_count": 128,
            "stratum_counts": counts,
            "sha256": state_hash,
            "frozen_at_utc": frozen_at,
            "freeze_completed_before_model_loading": True,
            "expected_labels_present": False,
        },
        "freshness_audit": audit,
        "seeds": {
            "review_set": REVIEW_SET_SEED,
            "model_blinding": MODEL_BLINDING_SEED,
        },
        "preprocessing": {
            "first_sequence": "Problem:\n<problem>",
            "second_sequence": "Conversation:\n<formatted dialogue>\n\nNext teacher pedagogical move:",
            "dialogue_format": "{user}: {text}",
            "truncation_side": "left",
            "truncation": "only_second",
            "max_length": 512,
            "move_order": list(MOVES),
            "frozen_implementation_sha256": before_hashes["frozen_inference"],
        },
        "models": {
            "reference": {
                "path": BASELINE.relative_to(ROOT).as_posix(),
                "hashes": baseline_hashes,
            },
            "recalibrated": {
                "path": CANDIDATE.relative_to(ROOT).as_posix(),
                "hashes": candidate_hashes,
            },
        },
        "preregistered_stopping": {
            "minimum_model_disagreements_to_review": MIN_MODEL_DISAGREEMENTS,
            "minimum_decisive_disagreements_for_non_D_decision": MIN_DECISIVE_DISAGREEMENTS,
            "packet": "all and only model top-1 disagreements; no agreement padding",
        },
        "protected_input_hashes_before": before_hashes,
        "safety": {
            "training": 0,
            "parameter_changes": 0,
            "promotion": 0,
            "production_changes": 0,
            "tutor_api_calls": 0,
            "bkt_updates": 0,
            "lints_updates_or_modifications": 0,
            "authoritative_real_data_writes": 0,
            "mathdial_final_test_use": 0,
            "mrbench_v3_test_use": 0,
        },
    }
    temp_manifest = OUT / "fresh_review_states_v2_manifest.json.tmp"
    temp_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temp_manifest, OUT / "fresh_review_states_v2_manifest.json")

    # Phase 2: model inference starts only after the state file is immutable by
    # protocol and its hash/preregistration/initial manifest exist.
    inference = load_frozen_inference()
    records = [
        {"problem": row["problem"], "history": row["dialogue_history"]}
        for row in frozen
    ]
    reference_vectors = inference.predict(BASELINE, records, batch_size=16)
    recalibrated_vectors = inference.predict(CANDIDATE, records, batch_size=16)
    if reference_vectors.shape != (128, 4) or recalibrated_vectors.shape != (128, 4):
        raise RuntimeError("Unexpected Review 2 prediction matrix shape")

    roles = ["reference", "recalibrated"]
    random.Random(MODEL_BLINDING_SEED).shuffle(roles)
    system_to_role = {"System X": roles[0], "System Y": roles[1]}
    mapping = {
        "access_classification": "SEALED_UNBLINDED_NOT_FOR_REVIEWER",
        "warning": "Do not provide this mapping to the Review 2 expert before judgments are frozen.",
        "seed": MODEL_BLINDING_SEED,
        "system_mapping": {
            system: {
                "model_role": role,
                "path": manifest["models"][role]["path"],
                "weight_sha256": manifest["models"][role]["hashes"]["model.safetensors"],
            }
            for system, role in system_to_role.items()
        },
    }
    (OUT / "hidden_model_mapping_v2.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    actual = {"reference": reference_vectors, "recalibrated": recalibrated_vectors}
    prediction_rows: list[dict[str, Any]] = []
    disagreement_ids: list[str] = []
    for index, state in enumerate(frozen):
        row: dict[str, Any] = {
            "access_classification": "SEALED_NOT_FOR_REVIEWER",
            "case_id": state["case_id"],
            "semantic_stratum": state["semantic_stratum"],
        }
        for system, role in system_to_role.items():
            prefix = system.casefold().replace(" ", "_")
            row.update(vector_fields(prefix, actual[role][index]))
        row["systems_agree"] = row["system_x_top1"] == row["system_y_top1"]
        row["primary_disagreement_packet"] = not row["systems_agree"]
        if not row["systems_agree"]:
            disagreement_ids.append(state["case_id"])
        prediction_rows.append(row)

    prediction_columns = [
        "access_classification", "case_id", "semantic_stratum", "systems_agree",
        "primary_disagreement_packet",
        *[f"system_x_p_{move}" for move in MOVES], "system_x_top1",
        *[f"system_y_p_{move}" for move in MOVES], "system_y_top1",
    ]
    with (OUT / "hidden_model_predictions_v2.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_columns)
        writer.writeheader()
        writer.writerows(prediction_rows)

    sufficient = len(disagreement_ids) >= MIN_MODEL_DISAGREEMENTS
    state_by_id = {row["case_id"]: row for row in frozen}
    packet_states = [state_by_id[case_id] for case_id in disagreement_ids] if sufficient else []
    instructions = reviewer_instructions(packet_states, state_hash)
    if not sufficient:
        instructions = f"""# Review 2 preparation stopped

The frozen pool produced {len(disagreement_ids)} model disagreements, fewer than the preregistered minimum of {MIN_MODEL_DISAGREEMENTS}. No expert packet is released and no model conclusion is permitted. Frozen state-set SHA-256: `{state_hash}`.
"""
    forbidden = (
        "md7-r1", "md7-r2", "candidate", "baseline", "system x", "system y",
        "system_x_p_", "system_y_p_", "semantic_stratum", "bkt", "mrb1",
        "evaluator result", "historical outcome", "review 1 expert",
    )
    lowered = instructions.casefold()
    leaked = [token for token in forbidden if token in lowered]
    if leaked:
        raise RuntimeError(f"Reviewer-facing instructions leak hidden information: {leaked}")
    (OUT / "expert_review_instructions_v2.md").write_text(
        instructions, encoding="utf-8"
    )
    with (OUT / "expert_review_form_v2.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case_id", "expert_move", "expert_confidence", "expert_reason"),
        )
        writer.writeheader()
        for case_id in disagreement_ids if sufficient else []:
            writer.writerow({
                "case_id": case_id,
                "expert_move": "",
                "expert_confidence": "",
                "expert_reason": "",
            })

    after_hashes = {name: sha256(path) for name, path in protected.items()}
    changes = {
        name: {"before": before_hashes[name], "after": after_hashes[name]}
        for name in protected
        if before_hashes[name] != after_hashes[name]
    }
    if changes:
        raise RuntimeError(f"Protected source changed during preparation: {changes}")

    agreement_count = sum(bool(row["systems_agree"]) for row in prediction_rows)
    status = (
        "EXPERT_REVIEW_PENDING"
        if sufficient
        else "STOPPED_INSUFFICIENT_MODEL_DISAGREEMENTS"
    )
    manifest.update({
        "status": status,
        "prediction_phase_started_after_freeze": True,
        "prediction_completed_at_utc": datetime.now(UTC).isoformat(),
        "comparison": {
            "agreement_count": agreement_count,
            "disagreement_count": len(disagreement_ids),
            "minimum_required": MIN_MODEL_DISAGREEMENTS,
            "sufficient_for_expert_packet": sufficient,
            "packet_count": len(packet_states),
            "agreement_padding_count": 0,
        },
        "artifacts": {
            "hidden_model_predictions_v2.csv": {
                "sha256": sha256(OUT / "hidden_model_predictions_v2.csv"),
                "classification": "SEALED_NOT_FOR_REVIEWER",
            },
            "hidden_model_mapping_v2.json": {
                "sha256": sha256(OUT / "hidden_model_mapping_v2.json"),
                "classification": "SEALED_UNBLINDED_NOT_FOR_REVIEWER",
            },
            "expert_review_instructions_v2.md": {
                "sha256": sha256(OUT / "expert_review_instructions_v2.md"),
                "classification": "REVIEWER_SAFE_BLINDED" if sufficient else "STOP_NOTICE",
            },
            "expert_review_form_v2.csv": {
                "sha256_at_creation": sha256(OUT / "expert_review_form_v2.csv"),
                "classification": "REVIEWER_SAFE_BLINDED_EMPTY_FORM",
                "expert_fields_completed": 0,
            },
        },
        "artifact_access_policy": {
            "reviewer_safe": ["expert_review_instructions_v2.md", "expert_review_form_v2.csv"],
            "sealed_not_for_reviewer": ["hidden_model_predictions_v2.csv", "hidden_model_mapping_v2.json"],
            "internal_not_for_reviewer": [
                "fresh_review_states_v2.jsonl", "fresh_review_states_v2_manifest.json",
                "preregistration.md", "prepare_review_v2.py", "analyze_review_v2.py",
            ],
        },
        "blinding_audit": {
            "reviewer_files_contain_model_identity_or_system_mapping": False,
            "reviewer_files_contain_probabilities": False,
            "reviewer_files_contain_semantic_strata": False,
            "reviewer_files_contain_outcomes_or_review1_labels": False,
            "opaque_case_ids": True,
            "all_disagreements_in_packet": sufficient,
            "agreement_padding_count": 0,
        },
        "protected_input_hashes_after": after_hashes,
        "protected_input_changes": changes,
    })
    temp_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temp_manifest, OUT / "fresh_review_states_v2_manifest.json")

    print(json.dumps({
        "status": status,
        "frozen_rows": 128,
        "stratum_counts": counts,
        "review_set_sha256": state_hash,
        "maximum_similarity": audit["maximum_similarity_to_any_excluded_prior_material"],
        "agreements": agreement_count,
        "disagreements": len(disagreement_ids),
        "expert_packet": len(packet_states),
        "expert_judgments_filled": 0,
        "final_decision": None,
    }, indent=2))


if __name__ == "__main__":
    main()
