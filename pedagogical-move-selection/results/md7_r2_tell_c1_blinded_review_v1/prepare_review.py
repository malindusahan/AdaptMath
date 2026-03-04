from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import re
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
V1 = ROOT / "pedagogical-move-selection" / "results" / "md7_telling_real_state_diagnostic_v1"
PREPARATION = ROOT / "pedagogical-move-selection" / "results" / "md7_r2_tell_c1_preparation_v1"
BASELINE = ROOT / "pedagogical-move-selection" / "models" / "candidates" / "md7r1_epoch3"
CANDIDATE = ROOT / "pedagogical-move-selection" / "models" / "candidates" / "md7_r2_tell_c1" / "epoch2"

MOVES = ("generic", "probing", "focus", "telling")
REVIEW_SET_SEED = 20260828
MODEL_BLINDING_SEED = 2026082801
PACKET_SELECTION_SEED = 2026082802
TARGET_PACKET_MIN_IF_FEW_DISAGREEMENTS = 24
EXPECTED_V1_ANALYSIS_HASH = "adb98cf8c978479d7aedb437c01bbbbe62e13b6bebd94acf5a932f0397dafd2c"
EXPECTED_BASELINE_WEIGHT = "d32d37f7f664d038f80fefb92f874804b49e5ac3d848985636ae64e6a2a60a13"
EXPECTED_CANDIDATE_WEIGHT = "ad770256a71efcce7485705099511494e41806b2e29bc3829f9ba8aeaf731cd5"

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


def case(
    draft_id: str,
    semantic_stratum: str,
    problem: str,
    dialogue_history: Sequence[tuple[str, str]],
    latest_student_state: str,
) -> dict[str, Any]:
    return {
        "draft_id": draft_id,
        "semantic_stratum": semantic_stratum,
        "problem": problem,
        "dialogue_history": [
            {"user": user, "text": text} for user, text in dialogue_history
        ],
        "latest_student_state": latest_student_state,
    }


# Authored independently as a balanced deployment-style set. These are not
# generated from model outputs and contain no expected pedagogical-move label.
DRAFT_CASES: list[dict[str, Any]] = [
    case(
        "A01", STRATA[0],
        "A paint mixture uses 5 cups of blue paint for every 8 cups of white paint. If 24 cups of white paint are used, how many cups of blue paint are needed?",
        [("teacher", "How would you set up the relationship between the blue and white paint?"),
         ("student", "I multiplied 24 by 8, so I think 192 cups of blue paint are needed.")],
        "The learner gives a clear first answer that reverses the ratio relationship.",
    ),
    case(
        "A02", STRATA[0],
        "A bag contains 7 red counters, 4 green counters, and 9 yellow counters. What is the probability of drawing a green counter at random?",
        [("teacher", "What probability do you get?"),
         ("student", "There are 4 green and 9 yellow, so I got 13/20.")],
        "The first answer uses an incorrect favorable-outcome count.",
    ),
    case(
        "A03", STRATA[0],
        "Find the slope of the line through (-3, 4) and (5, 12).",
        [("teacher", "Show how you would calculate the slope."),
         ("student", "I added the coordinates: (-3 + 5)/(4 + 12) = 2/16, so the slope is 1/8.")],
        "The learner's first calculation uses sums rather than coordinate differences.",
    ),
    case(
        "A04", STRATA[0],
        "A jacket costs $84 before a 25% discount. What is the sale price?",
        [("teacher", "What calculation would you use for the sale price?"),
         ("student", "I added 25 to 84 and got $109.")],
        "The learner treats the percent as a fixed amount on the first attempt.",
    ),
    case(
        "A05", STRATA[0],
        "A trapezoid has parallel sides of 11 cm and 17 cm and a height of 6 cm. Find its area.",
        [("teacher", "What area do you obtain?"),
         ("student", "I multiplied the two parallel sides, 11 times 17, and got 187 square centimetres.")],
        "The learner's first answer multiplies the bases and omits the height relationship.",
    ),
    case(
        "A06", STRATA[0],
        "Evaluate 18 - 3(4 + 2).",
        [("teacher", "Walk me through the order you used."),
         ("student", "I did 18 - 3 first to get 15, then 15 times 6, so my answer is 90.")],
        "The first solution performs subtraction before the grouped multiplication.",
    ),

    case(
        "B01", STRATA[1],
        "Write 0.0000725 in scientific notation.",
        [("teacher", "How would you begin writing this number in scientific notation?"),
         ("student", "I don't know where the decimal point should go.")],
        "This is the learner's first stated confusion about the task.",
    ),
    case(
        "B02", STRATA[1],
        "A kayak travels 18 kilometres in 2.5 hours at a constant rate. Find its speed in kilometres per hour.",
        [("teacher", "What would you do first to find the rate per hour?"),
         ("student", "I'm not sure which number should be divided by the other.")],
        "The learner expresses first-time uncertainty about the operation order.",
    ),
    case(
        "B03", STRATA[1],
        "Find the median of 6, 9, 12, 15, 18, and 27.",
        [("teacher", "How would you find the median of this list?"),
         ("student", "There are two numbers in the middle, and I don't know what to do then.")],
        "The learner has identified the source of first confusion but has received no scaffold.",
    ),
    case(
        "B04", STRATA[1],
        "An exterior angle of a triangle is 128 degrees. One remote interior angle is 53 degrees. Find the other remote interior angle.",
        [("teacher", "What relationship could help you find the missing angle?"),
         ("student", "I can't remember the rule for an exterior angle.")],
        "The learner reports a first recall gap.",
    ),
    case(
        "B05", STRATA[1],
        "A cube has an edge length of 4.5 cm. Find its volume.",
        [("teacher", "How would you start finding the cube's volume?"),
         ("student", "I have no idea which measurements I need to multiply.")],
        "The learner gives a first non-specific statement of confusion.",
    ),
    case(
        "B06", STRATA[1],
        "On a map, 3 cm represents 14 km. Two towns are 7.5 cm apart on the map. Find their actual distance.",
        [("teacher", "How might you use the map scale here?"),
         ("student", "I don't know how to turn 7.5 centimetres into the real distance.")],
        "The learner first expresses uncertainty about applying the scale.",
    ),

    case(
        "C01", STRATA[2],
        "Which is greater, 7/12 or 5/8? Explain your comparison.",
        [("teacher", "Which fraction do you think is greater?"),
         ("student", "I think 7/12 is greater because 7 is bigger than 5."),
         ("teacher", "Rewrite both fractions using a denominator of 24 so their parts have the same size."),
         ("student", "That gives 14/24 and 15/24, so I think 5/8 is the greater one now.")],
        "After one denominator scaffold, the learner constructs equivalent fractions and reverses the comparison.",
    ),
    case(
        "C02", STRATA[2],
        "Solve 4m - 9 = 31.",
        [("teacher", "What is your first step?"),
         ("student", "I divided 31 by 4 and got m = 7.75."),
         ("teacher", "Before dividing by 4, undo the subtraction of 9 on both sides."),
         ("student", "Then 4m = 40. I think I divide by 4 next, but I want to check.")],
        "One scaffold produces the correct intermediate equation and a plausible next operation.",
    ),
    case(
        "C03", STRATA[2],
        "A triangle has a base of 14 m and a perpendicular height of 9 m. Find its area.",
        [("teacher", "What area did you calculate?"),
         ("student", "I used 14 times 9 and got 126 square metres."),
         ("teacher", "A triangle with that base and height occupies half of the matching rectangle."),
         ("student", "So I should take half of 126, which would be 63 square metres?")],
        "The learner uses one geometric cue to propose the corrected area.",
    ),
    case(
        "C04", STRATA[2],
        "A box contains 5 blue and 7 orange tokens. Two tokens are drawn without replacement. What is the probability that both are blue?",
        [("teacher", "How would you write the two probabilities?"),
         ("student", "I wrote 5/12 times 5/12."),
         ("teacher", "After one blue token is drawn and not replaced, update both the blue count and the total count."),
         ("student", "Then the second probability should be 4/11, so the product starts 5/12 times 4/11.")],
        "One scaffold leads to the correct without-replacement structure.",
    ),
    case(
        "C05", STRATA[2],
        "A circular fountain has radius 6.5 m. Find its circumference in terms of pi.",
        [("teacher", "Which expression would you use?"),
         ("student", "I used pi times 6.5 squared."),
         ("teacher", "That expression measures the region inside the circle; choose the formula for the distance around it."),
         ("student", "The distance around is 2 pi r, so it should be 13 pi metres.")],
        "After one distinction between area and circumference, the learner gives the correct expression.",
    ),
    case(
        "C06", STRATA[2],
        "Convert 2.4 kilometres to metres.",
        [("teacher", "What conversion did you get?"),
         ("student", "I moved the decimal two places and got 240 metres."),
         ("teacher", "One kilometre contains 1,000 metres, so scale 2.4 by 1,000."),
         ("student", "That would move it three places, giving 2,400 metres, right?")],
        "The learner applies one unit-scale cue and reaches the corrected value.",
    ),

    case(
        "D01", STRATA[3],
        "Evaluate 2^(-3).",
        [("teacher", "What value do you get?"),
         ("student", "I get -8 because the exponent is negative."),
         ("teacher", "A negative exponent indicates a reciprocal; it does not make the base negative."),
         ("student", "I still think the negative sign makes it -8."),
         ("teacher", "Rewrite 2^(-3) as 1 divided by 2^3 before evaluating."),
         ("student", "I keep arriving at -8 when I calculate it.")],
        "The same sign misconception persists after two distinct scaffolds.",
    ),
    case(
        "D02", STRATA[3],
        "Solve the proportion 7/9 = n/36.",
        [("teacher", "How did you solve for n?"),
         ("student", "I added 7 and 36, so n = 43."),
         ("teacher", "The ratios are equal; identify the factor that changes 9 into 36."),
         ("student", "I added 27 to the denominator, so I should add 27 to 7 and get 34."),
         ("teacher", "Equivalent ratios multiply both numerator and denominator by the same factor."),
         ("student", "I still want to add 27, so my answer remains 34.")],
        "The additive conception of equivalent ratios repeats after two scaffolds.",
    ),
    case(
        "D03", STRATA[3],
        "Expand and simplify -3(2x - 5).",
        [("teacher", "What expression do you obtain?"),
         ("student", "I get -6x - 15."),
         ("teacher", "Distribute -3 to both terms and pay attention to the product of two negative signs."),
         ("student", "I distributed it and still got -6x - 15."),
         ("teacher", "Write the second product explicitly as (-3)(-5)."),
         ("student", "I think (-3)(-5) is -15, so I still have -6x - 15.")],
        "The learner repeats the negative-times-negative error after two focused prompts.",
    ),
    case(
        "D04", STRATA[3],
        "Calculate 5/6 - 1/4.",
        [("teacher", "Show how you subtract the fractions."),
         ("student", "I subtracted top and bottom and got 4/2."),
         ("teacher", "First rename both fractions using one common denominator."),
         ("student", "I used 6 - 4 = 2 as the common denominator, so I still get 4/2."),
         ("teacher", "Choose a denominator that is a multiple of both 6 and 4, then rewrite each fraction."),
         ("student", "I keep subtracting the denominators and getting 2.")],
        "The denominator-subtraction misconception remains after two scaffolds.",
    ),
    case(
        "D05", STRATA[3],
        "A town's population of 480 increases by 15%. Find the new population.",
        [("teacher", "What calculation did you use?"),
         ("student", "I added 15 and got 495 people."),
         ("teacher", "The increase is 15 percent of 480, not 15 people."),
         ("student", "I added the percent number to 480 again, so I still have 495."),
         ("teacher", "Find 0.15 times 480 before adding the increase to the original population."),
         ("student", "I still think increase by 15% means add 15.")],
        "The fixed-addition interpretation of percent increase persists after two explanations.",
    ),
    case(
        "D06", STRATA[3],
        "A survey records scores 2, 4, and 6 with frequencies 1, 3, and 2 respectively. Find the mean score.",
        [("teacher", "How did you calculate the mean?"),
         ("student", "I averaged 2, 4, and 6 and got 4."),
         ("teacher", "Each score must contribute as many times as its frequency indicates."),
         ("student", "I still added 2 + 4 + 6 and divided by 3."),
         ("teacher", "Form the weighted total 2(1) + 4(3) + 6(2), then divide by the total frequency."),
         ("student", "I keep using just the three listed scores, so I get 4 again.")],
        "The learner continues to ignore frequencies after two targeted scaffolds.",
    ),

    case(
        "E01", STRATA[4],
        "Solve the system y = 2x + 1 and 3x + y = 16.",
        [("teacher", "How could the first equation help with the second?"),
         ("student", "I don't know how to use the two equations together."),
         ("teacher", "Replace y in the second equation with the expression 2x + 1."),
         ("student", "I see y = 2x + 1, but I don't know where to put it."),
         ("teacher", "Write 3x + (2x + 1) = 16, then combine the x terms."),
         ("student", "I still cannot see how to continue from that equation.")],
        "The learner remains confused after two increasingly explicit scaffolds.",
    ),
    case(
        "E02", STRATA[4],
        "Calculate 6.72 divided by 0.8.",
        [("teacher", "How would you begin the division?"),
         ("student", "I don't know how to divide when the divisor has a decimal."),
         ("teacher", "Multiply both numbers by 10 so the divisor becomes a whole number."),
         ("student", "I changed 0.8 to 8, but I don't know what happens to 6.72."),
         ("teacher", "Apply the same factor to 6.72, giving 67.2 divided by 8."),
         ("student", "I still don't know how to carry out 67.2 divided by 8.")],
        "Confusion persists after the decimal transformation has been fully narrowed.",
    ),
    case(
        "E03", STRATA[4],
        "A right triangle has hypotenuse 17 cm and one leg 8 cm. Find the other leg.",
        [("teacher", "How would you set up the Pythagorean relationship?"),
         ("student", "I know the theorem, but I don't know where the 17 goes."),
         ("teacher", "The hypotenuse is c, so write 8^2 + b^2 = 17^2."),
         ("student", "I can copy that equation, but I don't know how to get b."),
         ("teacher", "Subtract 8^2 from 17^2, then take the square root."),
         ("student", "I'm still lost about doing those steps.")],
        "The learner remains unable to execute a narrowed two-step calculation.",
    ),
    case(
        "E04", STRATA[4],
        "Factor x^2 + 11x + 24.",
        [("teacher", "What factor pair are you looking for?"),
         ("student", "I don't know how to choose the two numbers."),
         ("teacher", "Look for two integers whose product is 24 and whose sum is 11."),
         ("student", "I listed factors of 24, but I still don't know which pair works."),
         ("teacher", "Check 3 and 8 against both the product and sum conditions."),
         ("student", "I still don't understand how those numbers become factors of the quadratic.")],
        "The learner remains confused after the specific factor pair has been identified.",
    ),
    case(
        "E05", STRATA[4],
        "A metal sample has mass 156 g and volume 20 cubic centimetres. Find its density.",
        [("teacher", "Which quantities belong in the density calculation?"),
         ("student", "I don't know whether to multiply or divide them."),
         ("teacher", "Density is mass divided by volume, so use 156 divided by 20."),
         ("student", "I know the formula now, but I don't know how to calculate that division."),
         ("teacher", "Break 156/20 into 140/20 plus 16/20."),
         ("student", "I'm still unable to combine those parts.")],
        "The learner remains stuck after two procedural scaffolds.",
    ),
    case(
        "E06", STRATA[4],
        "A game pays $0 with probability 0.5, $4 with probability 0.3, and $10 with probability 0.2. Find the expected payout.",
        [("teacher", "How would you combine the payouts and probabilities?"),
         ("student", "I don't know what expected payout means here."),
         ("teacher", "Multiply each payout by its probability, then add the products."),
         ("student", "I wrote 0 times 0.5, but I don't know what to do with the other rows."),
         ("teacher", "Also calculate 4 times 0.3 and 10 times 0.2 before adding all three results."),
         ("student", "I still can't put those calculations together.")],
        "Repeated confusion remains after the complete calculation structure is supplied.",
    ),

    case(
        "F01", STRATA[5],
        "Simplify a^7 / a^3 for nonzero a.",
        [("teacher", "How do the exponents change when equal bases are divided?"),
         ("student", "I think the answer is a^(7/3)."),
         ("teacher", "Division of equal bases uses the difference of their exponents."),
         ("student", "I tried subtracting but I am not sure why it works."),
         ("teacher", "Expand a^7 and cancel the three matching factors from the denominator."),
         ("student", "Could you explain the quotient rule directly and show how the cancellation gives the exponent?")],
        "After prior support, the learner explicitly requests a direct explanation of the rule.",
    ),
    case(
        "F02", STRATA[5],
        "Thirty percent of a number is 27. Find the number.",
        [("teacher", "What equation could represent the statement?"),
         ("student", "I wrote 30 + n = 27 and got stuck."),
         ("teacher", "Represent 30% as 0.30 multiplied by the unknown number."),
         ("student", "I wrote 0.30n = 27, but I don't understand how to isolate n."),
         ("teacher", "Think about undoing multiplication by 0.30."),
         ("student", "Please show me the method for solving this percent equation and explain each step.")],
        "The learner directly asks for an explanation after two supports.",
    ),
    case(
        "F03", STRATA[5],
        "Solve -4x > 28.",
        [("teacher", "What happens when you divide both sides by -4?"),
         ("student", "I got x > -7."),
         ("teacher", "Dividing an inequality by a negative reverses its direction."),
         ("student", "I changed the sign, but I don't understand why it must reverse."),
         ("teacher", "Compare a simple true inequality such as 2 < 5 after multiplying both sides by -1."),
         ("student", "Can you explain directly why the inequality flips and then apply it to this problem?")],
        "The learner explicitly requests an explanation after conceptual and example-based support.",
    ),
    case(
        "F04", STRATA[5],
        "Two similar triangles have corresponding sides 9 cm and 15 cm. A second side of the smaller triangle is 12 cm. Find the corresponding side of the larger triangle.",
        [("teacher", "How could you use the corresponding sides to form a scale factor?"),
         ("student", "I subtracted 15 - 9 and added 6 to 12, giving 18."),
         ("teacher", "Similarity uses a multiplicative scale factor rather than a fixed difference."),
         ("student", "I know I should use 15/9, but I am unsure how it acts on 12."),
         ("teacher", "The same multiplier that maps 9 to 15 must map 12 to its partner."),
         ("student", "Please explain the scale-factor method directly and work through the multiplication.")],
        "The learner requests a direct worked explanation after two meaningful scaffolds.",
    ),
    case(
        "F05", STRATA[5],
        "A spinner lands on red with probability 0.4 and blue with probability 0.6. It is spun twice. Find the probability of exactly one red result.",
        [("teacher", "What outcome paths give exactly one red?"),
         ("student", "I only used red then blue, so I wrote 0.4 times 0.6."),
         ("teacher", "Exactly one red can occur in two different orders."),
         ("student", "I can name red-blue and blue-red, but I don't know how to combine them."),
         ("teacher", "Find each path probability and then combine the mutually exclusive paths."),
         ("student", "Could you explain the probability-tree method and show why the two paths are added?")],
        "The learner explicitly requests explanation after both paths have been focused.",
    ),
    case(
        "F06", STRATA[5],
        "Rewrite x^2 + 6x + 1 in completed-square form.",
        [("teacher", "What value helps complete the square for x^2 + 6x?"),
         ("student", "I added 6 and wrote (x + 6)^2 + 1."),
         ("teacher", "Use half the coefficient of x, then square that half."),
         ("student", "Half is 3 and its square is 9, but I don't know how to keep the expression equivalent."),
         ("teacher", "Add and subtract the same 9 so the overall value does not change."),
         ("student", "Please show the completed-square steps directly and explain where the correction term goes.")],
        "The learner requests a direct method after two targeted supports.",
    ),

    case(
        "G01", STRATA[6],
        "Calculate 8.4 divided by 0.7.",
        [("teacher", "How would you handle the decimal divisor?"),
         ("student", "I first wrote 8.4/7 and got 1.2."),
         ("teacher", "Scale both dividend and divisor by 10 so the division stays equivalent."),
         ("student", "Then it becomes 84 divided by 7, which equals 12.")],
        "After earlier difficulty, the latest response correctly applies the scaling and completes the calculation.",
    ),
    case(
        "G02", STRATA[6],
        "Solve the system x + y = 11 and x - y = 3.",
        [("teacher", "What happens if you combine the two equations?"),
         ("student", "I subtracted them and got confused because both variables seemed to disappear."),
         ("teacher", "Add the equations term by term so the y and -y cancel."),
         ("student", "Adding gives 2x = 14, so x = 7. Then y must be 4 because 7 + 4 = 11.")],
        "The latest response shows a complete and checked recovery after one targeted scaffold.",
    ),
    case(
        "G03", STRATA[6],
        "Find the surface area of a rectangular prism with length 8 cm, width 3 cm, and height 5 cm.",
        [("teacher", "How did you account for all the faces?"),
         ("student", "I multiplied 8 times 3 times 5 and got 120 square centimetres."),
         ("teacher", "Volume multiplies all three dimensions; surface area adds the areas of the three pairs of opposite faces."),
         ("student", "So I use 2(8 times 3) + 2(8 times 5) + 2(3 times 5) = 48 + 80 + 30 = 158 square centimetres.")],
        "The learner corrects the volume-area confusion and completes all face calculations.",
    ),
    case(
        "G04", STRATA[6],
        "An account starts with $600 and earns 4% compound interest annually for 3 years. Find its value after 3 years.",
        [("teacher", "How would you represent the repeated annual growth?"),
         ("student", "I added 4 three times and wrote 600 + 12."),
         ("teacher", "Use a growth factor of 1.04 once for each year."),
         ("student", "Then the value is 600(1.04)^3. I calculate (1.04)^3 first and then multiply by 600.")],
        "The latest response correctly establishes the compound-growth method after difficulty.",
    ),
    case(
        "G05", STRATA[6],
        "A jar has 6 black and 5 white beads. Two beads are drawn without replacement. Find the probability of drawing a black bead followed by a white bead.",
        [("teacher", "How did you represent the second draw?"),
         ("student", "I used 6/11 times 5/11 because the jar started with 11 beads."),
         ("teacher", "After the first bead is removed, update the total for the second draw."),
         ("student", "The first probability is 6/11, and after removing a black bead there are 5 white among 10 total, so I use 6/11 times 5/10.")],
        "The learner correctly repairs the conditional denominator and explains the remaining counts.",
    ),
    case(
        "G06", STRATA[6],
        "Find the midpoint of the segment joining (-6, 7) and (10, -1).",
        [("teacher", "How did you find the midpoint?"),
         ("student", "I subtracted the coordinates and got (-16, 8)."),
         ("teacher", "A midpoint averages the two x-coordinates and separately averages the two y-coordinates."),
         ("student", "The averages are (-6 + 10)/2 = 2 and (7 + -1)/2 = 3, so the midpoint is (2, 3).")],
        "The latest response fully corrects the coordinate operation and obtains the midpoint.",
    ),

    case(
        "H01", STRATA[7],
        "Use the quadratic formula to solve 2x^2 - 5x - 3 = 0.",
        [("teacher", "Which values are a, b, and c?"),
         ("student", "I used a = 2, b = 5, and c = 3, and my discriminant did not work."),
         ("teacher", "Keep the signs attached to the coefficients when identifying b and c."),
         ("student", "Then a = 2, b = -5, and c = -3. I think the numerator begins -(-5) plus or minus the square root, but I have not finished it.")],
        "The learner repairs the coefficient signs and partially sets up the formula.",
    ),
    case(
        "H02", STRATA[7],
        "A cyclist rides 27 km in 1.5 hours. At the same rate, how far will the cyclist travel in 4 hours?",
        [("teacher", "How could you find the distance for four hours?"),
         ("student", "I multiplied 27 by 1.5 and got 40.5, but that does not use the four hours."),
         ("teacher", "First find the distance travelled in one hour, then scale that unit rate to four hours."),
         ("student", "The unit rate should be 27 divided by 1.5. I think that is 18 km per hour, and then I still need to use the 4.")],
        "The learner recovers the unit-rate structure and value but has not completed the final multiplication.",
    ),
    case(
        "H03", STRATA[7],
        "A circle has radius 9 cm. Find the area of a 140-degree sector in terms of pi.",
        [("teacher", "How did you account for the sector angle?"),
         ("student", "I used 140 times pi times 9 squared."),
         ("teacher", "A sector is the fraction angle/360 of the full circle's area."),
         ("student", "So I should use 140/360 times 81pi. I can simplify 140/360, but I am not sure how far to reduce it.")],
        "The learner now has the correct sector structure and needs only local simplification.",
    ),
    case(
        "H04", STRATA[7],
        "A histogram class from 20 to 30 contains 45 observations. Find its frequency density.",
        [("teacher", "How did you calculate the frequency density?"),
         ("student", "I multiplied 45 by 30 and got 1,350."),
         ("teacher", "Frequency density divides frequency by the class width, and the width is the upper boundary minus the lower boundary."),
         ("student", "The class width is 30 - 20 = 10, so I should calculate 45 divided by 10. I think that gives 4.5.")],
        "The learner recovers both the class width and division, with tentative correct arithmetic.",
    ),
    case(
        "H05", STRATA[7],
        "Three consecutive integers have a sum of 96. Find the integers.",
        [("teacher", "How could you represent the three consecutive integers?"),
         ("student", "I called all three numbers n, so I wrote n + n + n = 96."),
         ("teacher", "If the first integer is n, express the next two as one more and two more than n."),
         ("student", "Then the equation is n + (n + 1) + (n + 2) = 96. I combine that to 3n + 3 = 96, but I have not solved for n yet.")],
        "The learner corrects the representation and simplifies the equation but has not isolated the variable.",
    ),
    case(
        "H06", STRATA[7],
        "Reflect the triangle with vertices (2, 1), (5, 1), and (3, 4) across the y-axis.",
        [("teacher", "What happens to each coordinate in a reflection across the y-axis?"),
         ("student", "I changed both signs and got (-2, -1) for the first point."),
         ("teacher", "Across the y-axis, only the x-coordinate changes sign; the y-coordinate stays fixed."),
         ("student", "Then (2, 1) becomes (-2, 1). I think I do the same x-sign change to the other two vertices.")],
        "The learner correctly applies the transformation to one point and states the rule for the remaining points.",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def previous_state_records() -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}

    with (V1 / "telling_boundary_challenge.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    records["prior_manual_48"] = [
        {"problem": row["problem"], "history": json.loads(row["history"])}
        for row in rows
    ]

    with (V1 / "real_state_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        real_rows = list(csv.DictReader(handle))
    records["prior_deployment_87"] = [
        {
            "problem": row["problem"],
            "history": json.loads(row["history_before_action"]),
        }
        for row in real_rows
    ]

    for key, filename in (
        ("c1_correction_train", "correction_train.jsonl"),
        ("calibration_train", "telling_calibration_train_v1.jsonl"),
        ("calibration_validation", "telling_calibration_validation_v1.jsonl"),
        ("calibration_challenge", "telling_boundary_challenge_val_v1.jsonl"),
    ):
        records[key] = [
            {"problem": row["problem"], "history": row.get("history", "")}
            for row in read_jsonl(PREPARATION / filename)
        ]
    return records


def freeze_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(DRAFT_CASES) != 48:
        raise AssertionError(f"Expected 48 authored cases, found {len(DRAFT_CASES)}")
    if Counter(row["semantic_stratum"] for row in DRAFT_CASES) != Counter(
        {name: 6 for name in STRATA}
    ):
        raise AssertionError("Authored stratum balance is not exactly 6 per stratum")
    if len({row["draft_id"] for row in DRAFT_CASES}) != 48:
        raise AssertionError("Draft IDs must be unique")

    prior = previous_state_records()
    prior_fingerprints = {
        source: {fingerprint(row["problem"], row["history"]) for row in rows}
        for source, rows in prior.items()
    }

    exact_overlap: dict[str, int] = {}
    for source, known in prior_fingerprints.items():
        exact_overlap[source] = sum(
            fingerprint(row["problem"], row["dialogue_history"]) in known
            for row in DRAFT_CASES
        )
    if any(exact_overlap.values()):
        raise AssertionError(f"Fresh-state exact overlap detected: {exact_overlap}")

    # The anti-paraphrase audit is deliberately against the prior hand-authored
    # 48, not protected tests or labels. It is a lexical warning boundary, not
    # a model-selection criterion.
    prior_48_text = [
        fingerprint(row["problem"], row["history"])
        for row in prior["prior_manual_48"]
    ]
    nearest: list[dict[str, Any]] = []
    for row in DRAFT_CASES:
        current = fingerprint(row["problem"], row["dialogue_history"])
        ratios = [SequenceMatcher(None, current, known).ratio() for known in prior_48_text]
        best_index = int(np.argmax(ratios))
        nearest.append(
            {
                "draft_id": row["draft_id"],
                "prior_case_index": best_index,
                "sequence_similarity": float(ratios[best_index]),
            }
        )
    max_similarity = max(item["sequence_similarity"] for item in nearest)
    if max_similarity >= 0.80:
        raise AssertionError(
            f"A fresh case is too lexically close to the prior manual 48: {max_similarity}"
        )

    rng = random.Random(REVIEW_SET_SEED)
    shuffled = list(DRAFT_CASES)
    rng.shuffle(shuffled)
    frozen: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for row in shuffled:
        opaque = hashlib.sha256(
            f"{REVIEW_SET_SEED}:{row['draft_id']}".encode("utf-8")
        ).hexdigest()[:8].upper()
        case_id = f"REV-{opaque}"
        if case_id in used_ids:
            raise AssertionError("Opaque review ID collision")
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
    audit = {
        "authorship": "fresh_independently_authored_deployment_style_synthetic",
        "prior_48_used_as_case_source": False,
        "prior_48_used_for_exact_and_anti_paraphrase_audit_only": True,
        "correction_and_synthetic_artifacts_used_for_exact_overlap_audit_only": True,
        "exact_overlap_counts": exact_overlap,
        "maximum_sequence_similarity_to_prior_manual_48": max_similarity,
        "nearest_prior_manual_case_audit": nearest,
        "mathdial_final_test_read": False,
        "mrbench_v3_test_read": False,
        "outcome_signals_read_or_used": False,
    }
    return frozen, audit


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def load_frozen_v1():
    spec = importlib.util.spec_from_file_location("frozen_blinded_review_v1", V1 / "analysis.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen v1 diagnostic inference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_model(model_dir: Path, expected_weight: str) -> dict[str, str]:
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete model export {model_dir}: {missing}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if {int(key): value for key, value in config["id2label"].items()} != dict(enumerate(MOVES)):
        raise ValueError("Model label order mismatch")
    hashes = {name: sha256(model_dir / name) for name in required}
    if hashes["model.safetensors"] != expected_weight:
        raise ValueError("Model weight hash mismatch")
    return hashes


def vector_fields(prefix: str, vector: np.ndarray) -> dict[str, Any]:
    return {
        **{f"{prefix}_p_{move}": float(vector[index]) for index, move in enumerate(MOVES)},
        f"{prefix}_top1": MOVES[int(vector.argmax())],
    }


def choose_packet(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    disagreements = [row for row in rows if not row["systems_agree"]]
    agreements = [row for row in rows if row["systems_agree"]]
    selected = list(disagreements)
    supplementation: list[dict[str, Any]] = []
    if len(disagreements) < 20:
        needed = max(0, TARGET_PACKET_MIN_IF_FEW_DISAGREEMENTS - len(disagreements))
        rng = random.Random(PACKET_SELECTION_SEED)
        by_stratum: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATA}
        for row in agreements:
            by_stratum[row["semantic_stratum"]].append(row)
        for values in by_stratum.values():
            rng.shuffle(values)
        while needed and any(by_stratum.values()):
            for stratum in STRATA:
                if needed == 0:
                    break
                if by_stratum[stratum]:
                    supplementation.append(by_stratum[stratum].pop())
                    needed -= 1
        selected.extend(supplementation)
    # Forty-eight total cases are not impractical, so all disagreements are
    # retained even when there are more than 32.
    packet_rng = random.Random(PACKET_SELECTION_SEED)
    packet_rng.shuffle(selected)
    return [row["case_id"] for row in selected], {
        "rule": "all disagreements; if fewer than 20, deterministic stratum-round-robin agreement supplementation to 24; all disagreements retained if above 32",
        "disagreement_count": len(disagreements),
        "agreement_supplement_count": len(supplementation),
        "packet_count": len(selected),
        "selection_seed": PACKET_SELECTION_SEED,
    }


def review_instructions(
    packet_states: Sequence[dict[str, Any]],
    review_set_hash: str,
) -> str:
    text = f"""# Blinded Pedagogical-Move Expert Review

## Reviewer protocol

Review all cases independently using only the problem and conversation shown. For each case, choose exactly one most appropriate next pedagogical move, record confidence, and optionally give a one-sentence reason. Do not seek additional learner records or outcome information.

Use these fixed definitions for every case:

- **generic:** neutral encouragement, transition, or broadly supportive response without targeting a specific reasoning gap.
- **probing:** ask the learner to explain, reason, recall, or reveal their thinking.
- **focus:** direct attention to a specific error, relationship, clue, or sub-step without directly giving the explanation or solution.
- **telling:** provide explicit instruction, explanation, method, or missing step when further equivalent probing or focusing is unlikely to help.

Confidence:

- `1` = low
- `2` = medium
- `3` = high

Enter judgments only in `expert_review_form.csv`. Leave `expert_reason` empty or use no more than one sentence. Do not alter case IDs or add rows.

Packet integrity reference: frozen state-set SHA-256 `{review_set_hash}`. This material contains {len(packet_states)} reviewed cases.

## Review cases

"""
    for index, row in enumerate(packet_states, start=1):
        dialogue = "\n".join(
            f"{turn['user']}: {turn['text']}" for turn in row["dialogue_history"]
        )
        text += f"""### Case {index}: {row['case_id']}

**Problem**

{row['problem']}

**Conversation/history**

```text
{dialogue}
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

"""
    return text


def preregistration_text(
    review_set_hash: str,
    stratum_counts: dict[str, int],
    overlap_audit: dict[str, Any],
    baseline_hashes: dict[str, str],
    candidate_hashes: dict[str, str],
) -> str:
    return f"""# MD7-R2-TELL-C1 Blinded Expert Review Preregistration

Status: **PREPARED BEFORE MODEL COMPARISON; EXPERT LABELS NOT YET COLLECTED**

## Scientific question

Does the recalibrated selector improve pedagogical-move selection relative to the frozen reference around difficult telling-boundary states without introducing inappropriate telling after learner recovery or productive progress?

## Frozen review set

- Rows: 48 fresh independently authored deployment-style synthetic states.
- Strata: eight fixed semantic strata with six states each: `{json.dumps(stratum_counts, sort_keys=True)}`.
- SHA-256: `{review_set_hash}`.
- The prior 48-case diagnostic was not reused as a case source and the new set was not produced by one-by-one paraphrase.
- Exact state overlap against the prior 48, all 87 prior deployment states, the 2,400-row correction set, and packaged calibration train/validation/challenge artifacts: `{json.dumps(overlap_audit['exact_overlap_counts'], sort_keys=True)}`.
- Maximum normalized sequence similarity to any prior manually authored 48-case state: `{overlap_audit['maximum_sequence_similarity_to_prior_manual_48']:.6f}`; preregistered rejection boundary: `>= 0.80`.
- No expected move labels appear in the frozen review set.

## Fixed inference contract

Both systems are scored only after the set is frozen. Paired input is `Problem:\n<problem>` and `Conversation:\n<formatted dialogue>\n\nNext teacher pedagogical move:`. Dialogue uses `user: text`; tokenizer truncation side is left; truncation is `only_second`; maximum length is 512; move order is `generic, probing, focus, telling`.

Reference weight SHA-256: `{baseline_hashes['model.safetensors']}`. Recalibrated weight SHA-256: `{candidate_hashes['model.safetensors']}`.

## Blinding and packet selection

- Model identities are deterministically randomized to System X/Y with seed `{MODEL_BLINDING_SEED}` and stored only in `hidden_model_mapping.json`.
- Predictions and probabilities are stored only in `hidden_model_predictions.csv`.
- Human-facing files are `expert_review_instructions.md` and `expert_review_form.csv`; neither contains model identity, predictions, probabilities, strata, latest-state annotations, BKT, evaluator outcomes, MRB1, or historical outcomes.
- Include every disagreement. If there are fewer than 20 disagreements, add agreement states by deterministic stratum-round-robin sampling to reach 24, seed `{PACKET_SELECTION_SEED}`. If there are more than 32 disagreements, retain all because at most 48 is auditable.
- Reviewers choose exactly one move, confidence 1-3, and an optional one-sentence justification.

## Preregistered post-review metrics

For each actual model: exact agreement count/rate, four-class Macro-F1, per-class F1, telling precision/recall, and confusion counts. On model disagreements: recalibrated-only wins, reference-only wins, both wrong, decisive count, and a two-sided exact binomial test on recalibrated-vs-reference wins. Raw counts are primary; small-sample p-values will not be overstated.

Telling-specific reports are fixed for: expert-telling recall; false telling on expert-non-telling cases; G/H correct-or-recovering progress; C one-scaffold recovery; and D/E/F persistent/support-exhausted difficulty.

## Preregistered decision rule after expert labels

Integrity is checked first. Decision **D (REVIEW INCONCLUSIVE)** applies if the frozen hash fails, forms are incomplete, fewer than 8 decisive disagreements exist, median expert confidence is below 2, or the reviewed labels contain no telling or no non-telling case.

Otherwise, define a substantive G/H recovery overtrigger as either (i) at least 2 more false-telling cases for the recalibrated model than the reference among expert-non-telling G/H cases, or (ii) a false-telling-rate increase of at least 0.20 there. If this occurs, choose **B (RESIDUAL TELLING OVERCORRECTION)**.

If there is no substantive G/H overtrigger, choose **A (ADVANCE TO CONTROLLED LIVE VALIDATION)** only when the recalibrated model has strictly more disagreement wins than the reference and its telling precision is no more than 0.05 below the reference. Otherwise choose **C (TELLING BENEFIT NOT CONFIRMED)**.

No outcome authorizes production promotion.

## Safety boundary

Training 0; promotion 0; production changes 0; Tutor API calls 0; BKT updates 0; LinTS modifications/updates 0; authoritative real-data JSONL writes 0; MathDial final test use 0; MRBench V3 test use 0.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    completed_artifacts = (
        OUT / "preregistration.md",
        OUT / "fresh_review_states.jsonl",
        OUT / "fresh_review_states_manifest.json",
        OUT / "hidden_model_predictions.csv",
        OUT / "hidden_model_mapping.json",
        OUT / "expert_review_form.csv",
        OUT / "expert_review_instructions.md",
    )
    if all(path.is_file() for path in completed_artifacts):
        existing_manifest = json.loads(
            (OUT / "fresh_review_states_manifest.json").read_text(encoding="utf-8")
        )
        existing_hash = sha256(OUT / "fresh_review_states.jsonl")
        if existing_hash != existing_manifest["review_set"]["sha256"]:
            raise RuntimeError(
                "Refusing to rerun: the frozen review set no longer matches its manifest"
            )
        print(
            json.dumps(
                {
                    "status": "ALREADY_PREPARED_NO_FILES_CHANGED",
                    "review_set_sha256": existing_hash,
                    "expert_packet": existing_manifest["comparison"]["packet_selection"]["packet_count"],
                    "instruction": "Do not rerun preparation; complete expert_review_form.csv, then run analyze_review.py.",
                },
                indent=2,
            )
        )
        return

    protected = {
        "v1_analysis": V1 / "analysis.py",
        "prior_manual_48": V1 / "telling_boundary_challenge.csv",
        "prior_deployment_87": V1 / "real_state_predictions.csv",
        "c1_correction_train": PREPARATION / "correction_train.jsonl",
        "calibration_train": PREPARATION / "telling_calibration_train_v1.jsonl",
        "calibration_validation": PREPARATION / "telling_calibration_validation_v1.jsonl",
        "calibration_challenge": PREPARATION / "telling_boundary_challenge_val_v1.jsonl",
        "baseline_weight": BASELINE / "model.safetensors",
        "candidate_weight": CANDIDATE / "model.safetensors",
    }
    before_hashes = {name: sha256(path) for name, path in protected.items()}
    if before_hashes["v1_analysis"] != EXPECTED_V1_ANALYSIS_HASH:
        raise ValueError("Frozen v1 inference implementation hash mismatch")
    baseline_hashes = verify_model(BASELINE, EXPECTED_BASELINE_WEIGHT)
    candidate_hashes = verify_model(CANDIDATE, EXPECTED_CANDIDATE_WEIGHT)

    # PHASE 1: author/audit/freeze/hash. No model module has been imported and
    # no prediction has been generated before this block completes.
    frozen_states, overlap_audit = freeze_cases()
    states_path = OUT / "fresh_review_states.jsonl"
    write_jsonl(states_path, frozen_states)
    review_set_hash = sha256(states_path)
    stratum_counts = dict(Counter(row["semantic_stratum"] for row in frozen_states))
    frozen_at = datetime.now(UTC).isoformat()

    preregistration = preregistration_text(
        review_set_hash,
        stratum_counts,
        overlap_audit,
        baseline_hashes,
        candidate_hashes,
    )
    (OUT / "preregistration.md").write_text(preregistration, encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": "md7_r2_tell_c1_blinded_review_manifest_v1",
        "status": "frozen_before_model_comparison_expert_review_pending",
        "review_set": {
            "path": "fresh_review_states.jsonl",
            "access_classification": "internal_not_for_reviewer",
            "row_count": len(frozen_states),
            "stratum_counts": stratum_counts,
            "sha256": review_set_hash,
            "frozen_at_utc": frozen_at,
            "freeze_completed_before_model_loading": True,
            "expected_labels_present": False,
        },
        "overlap_audit": overlap_audit,
        "seeds": {
            "review_set": REVIEW_SET_SEED,
            "model_blinding": MODEL_BLINDING_SEED,
            "packet_selection": PACKET_SELECTION_SEED,
        },
        "preprocessing": {
            "first_sequence": "Problem:\n<problem>",
            "second_sequence": "Conversation:\n<formatted dialogue>\n\nNext teacher pedagogical move:",
            "dialogue_format": "{user}: {text}",
            "truncation_side": "left",
            "truncation": "only_second",
            "max_length": 512,
            "move_order": list(MOVES),
        },
        "models": {
            "reference": {"path": str(BASELINE.relative_to(ROOT)).replace("\\", "/"), "hashes": baseline_hashes},
            "recalibrated": {"path": str(CANDIDATE.relative_to(ROOT)).replace("\\", "/"), "hashes": candidate_hashes},
        },
        "protected_input_hashes_before": before_hashes,
        "safety": {
            "training": 0,
            "promotion": 0,
            "production_changes": 0,
            "tutor_api_calls": 0,
            "bkt_updates": 0,
            "lints_updates_or_modifications": 0,
            "authoritative_real_data_jsonl_writes": 0,
            "mathdial_final_test_use": 0,
            "mrbench_v3_test_use": 0,
        },
        "artifact_access_policy": {
            "reviewer_safe": ["expert_review_instructions.md", "expert_review_form.csv"],
            "sealed_not_for_reviewer": ["hidden_model_predictions.csv", "hidden_model_mapping.json"],
            "internal_not_for_reviewer": [
                "fresh_review_states.jsonl",
                "fresh_review_states_manifest.json",
                "preregistration.md",
                "prepare_review.py",
                "analyze_review.py",
            ],
        },
    }
    (OUT / "fresh_review_states_manifest.json.tmp").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(
        OUT / "fresh_review_states_manifest.json.tmp",
        OUT / "fresh_review_states_manifest.json",
    )

    # PHASE 2: only after the frozen file, hash, manifest, and preregistration
    # exist do we import the inference implementation and score the two models.
    v1 = load_frozen_v1()
    records = [
        {"problem": row["problem"], "history": row["dialogue_history"]}
        for row in frozen_states
    ]
    reference_probabilities = v1.predict(BASELINE, records, batch_size=16)
    recalibrated_probabilities = v1.predict(CANDIDATE, records, batch_size=16)
    if reference_probabilities.shape != (48, 4) or recalibrated_probabilities.shape != (48, 4):
        raise RuntimeError("Unexpected prediction matrix shape")

    model_order = ["reference", "recalibrated"]
    random.Random(MODEL_BLINDING_SEED).shuffle(model_order)
    system_to_model = {"System X": model_order[0], "System Y": model_order[1]}
    mapping = {
        "access_classification": "SEALED_UNBLINDED_NOT_FOR_REVIEWER",
        "warning": "Do not provide this file to reviewers before expert_review_form.csv is frozen.",
        "seed": MODEL_BLINDING_SEED,
        "system_mapping": {
            system: {
                "model_role": role,
                "path": manifest["models"][role]["path"],
                "weight_sha256": manifest["models"][role]["hashes"]["model.safetensors"],
            }
            for system, role in system_to_model.items()
        },
    }
    (OUT / "hidden_model_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    actual_vectors = {
        "reference": reference_probabilities,
        "recalibrated": recalibrated_probabilities,
    }
    prediction_rows: list[dict[str, Any]] = []
    for index, state in enumerate(frozen_states):
        row: dict[str, Any] = {
            "access_classification": "NOT_FOR_REVIEWER",
            "case_id": state["case_id"],
            "semantic_stratum": state["semantic_stratum"],
        }
        for system, role in system_to_model.items():
            prefix = system.casefold().replace(" ", "_")
            row.update(vector_fields(prefix, actual_vectors[role][index]))
        row["systems_agree"] = row["system_x_top1"] == row["system_y_top1"]
        prediction_rows.append(row)

    packet_ids, packet_selection = choose_packet(prediction_rows)
    packet_set = set(packet_ids)
    for row in prediction_rows:
        row["in_expert_packet"] = row["case_id"] in packet_set

    prediction_columns = [
        "access_classification", "case_id", "semantic_stratum", "systems_agree", "in_expert_packet",
        *[f"system_x_p_{move}" for move in MOVES], "system_x_top1",
        *[f"system_y_p_{move}" for move in MOVES], "system_y_top1",
    ]
    with (OUT / "hidden_model_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_columns)
        writer.writeheader()
        writer.writerows(prediction_rows)

    state_by_id = {row["case_id"]: row for row in frozen_states}
    packet_states = [state_by_id[case_id] for case_id in packet_ids]
    instructions = review_instructions(packet_states, review_set_hash)
    forbidden_human_tokens = (
        "md7-r1", "md7-r2", "candidate", "baseline", "system x", "system y",
        "system_x_p_", "system_y_p_", "p_generic", "p_probing", "p_focus",
        "p_telling", "semantic_stratum", "bkt", "mrb1", "evaluator category",
    )
    lowered_instructions = instructions.casefold()
    leaked = [token for token in forbidden_human_tokens if token in lowered_instructions]
    if leaked:
        raise RuntimeError(f"Reviewer instructions leak hidden information: {leaked}")
    (OUT / "expert_review_instructions.md").write_text(instructions, encoding="utf-8")

    with (OUT / "expert_review_form.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case_id", "expert_move", "expert_confidence", "expert_reason"),
        )
        writer.writeheader()
        for case_id in packet_ids:
            writer.writerow(
                {"case_id": case_id, "expert_move": "", "expert_confidence": "", "expert_reason": ""}
            )

    after_hashes = {name: sha256(path) for name, path in protected.items()}
    protected_changes = {
        name: {"before": before_hashes[name], "after": after_hashes[name]}
        for name in protected
        if before_hashes[name] != after_hashes[name]
    }
    if protected_changes:
        raise RuntimeError(f"Protected inputs changed: {protected_changes}")

    agreement_count = sum(bool(row["systems_agree"]) for row in prediction_rows)
    disagreement_count = len(prediction_rows) - agreement_count
    manifest.update(
        {
            "prediction_completed_at_utc": datetime.now(UTC).isoformat(),
            "prediction_phase_started_after_freeze": True,
            "comparison": {
                "agreement_count": agreement_count,
                "disagreement_count": disagreement_count,
                "packet_selection": packet_selection,
            },
            "artifacts": {
                "hidden_model_mapping.json": {
                    "sha256": sha256(OUT / "hidden_model_mapping.json"),
                    "classification": "SEALED_UNBLINDED_NOT_FOR_REVIEWER",
                },
                "hidden_model_predictions.csv": {
                    "sha256": sha256(OUT / "hidden_model_predictions.csv"),
                    "classification": "NOT_FOR_REVIEWER",
                },
                "expert_review_instructions.md": {
                    "sha256": sha256(OUT / "expert_review_instructions.md"),
                    "classification": "REVIEWER_SAFE_BLINDED",
                },
                "expert_review_form.csv": {
                    "sha256_at_creation": sha256(OUT / "expert_review_form.csv"),
                    "classification": "REVIEWER_SAFE_BLINDED_EMPTY_FORM",
                    "expert_fields_completed": 0,
                },
            },
            "blinding_audit": {
                "reviewer_files_contain_model_columns": False,
                "reviewer_files_contain_probabilities": False,
                "reviewer_files_contain_semantic_strata": False,
                "reviewer_files_contain_outcomes": False,
                "opaque_case_ids": True,
                "mapping_sealed_separately": True,
            },
            "protected_input_hashes_after": after_hashes,
            "protected_input_changes": protected_changes,
        }
    )
    (OUT / "fresh_review_states_manifest.json.tmp").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(
        OUT / "fresh_review_states_manifest.json.tmp",
        OUT / "fresh_review_states_manifest.json",
    )

    print(
        json.dumps(
            {
                "status": "PREPARED_EXPERT_REVIEW_PENDING",
                "frozen_states": len(frozen_states),
                "stratum_counts": stratum_counts,
                "review_set_sha256": review_set_hash,
                "agreements": agreement_count,
                "disagreements": disagreement_count,
                "expert_packet": len(packet_ids),
                "expert_judgments_filled": 0,
                "final_decision": None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
