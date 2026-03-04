# Blinded Pedagogical-Move Expert Review

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

Packet integrity reference: frozen state-set SHA-256 `58466681a7e40724f6b4c3acea28ac26c548a771872b9ed72d938dbe84353309`. This material contains 24 reviewed cases.

## Review cases

### Case 1: REV-E4908423

**Problem**

Rewrite x^2 + 6x + 1 in completed-square form.

**Conversation/history**

```text
teacher: What value helps complete the square for x^2 + 6x?
student: I added 6 and wrote (x + 6)^2 + 1.
teacher: Use half the coefficient of x, then square that half.
student: Half is 3 and its square is 9, but I don't know how to keep the expression equivalent.
teacher: Add and subtract the same 9 so the overall value does not change.
student: Please show the completed-square steps directly and explain where the correction term goes.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 2: REV-6BE163F8

**Problem**

Expand and simplify -3(2x - 5).

**Conversation/history**

```text
teacher: What expression do you obtain?
student: I get -6x - 15.
teacher: Distribute -3 to both terms and pay attention to the product of two negative signs.
student: I distributed it and still got -6x - 15.
teacher: Write the second product explicitly as (-3)(-5).
student: I think (-3)(-5) is -15, so I still have -6x - 15.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 3: REV-608EB56C

**Problem**

Find the slope of the line through (-3, 4) and (5, 12).

**Conversation/history**

```text
teacher: Show how you would calculate the slope.
student: I added the coordinates: (-3 + 5)/(4 + 12) = 2/16, so the slope is 1/8.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 4: REV-E13F202A

**Problem**

Calculate 6.72 divided by 0.8.

**Conversation/history**

```text
teacher: How would you begin the division?
student: I don't know how to divide when the divisor has a decimal.
teacher: Multiply both numbers by 10 so the divisor becomes a whole number.
student: I changed 0.8 to 8, but I don't know what happens to 6.72.
teacher: Apply the same factor to 6.72, giving 67.2 divided by 8.
student: I still don't know how to carry out 67.2 divided by 8.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 5: REV-5DCC1708

**Problem**

Which is greater, 7/12 or 5/8? Explain your comparison.

**Conversation/history**

```text
teacher: Which fraction do you think is greater?
student: I think 7/12 is greater because 7 is bigger than 5.
teacher: Rewrite both fractions using a denominator of 24 so their parts have the same size.
student: That gives 14/24 and 15/24, so I think 5/8 is the greater one now.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 6: REV-55EA762E

**Problem**

Solve 4m - 9 = 31.

**Conversation/history**

```text
teacher: What is your first step?
student: I divided 31 by 4 and got m = 7.75.
teacher: Before dividing by 4, undo the subtraction of 9 on both sides.
student: Then 4m = 40. I think I divide by 4 next, but I want to check.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 7: REV-ED5B6D23

**Problem**

A circular fountain has radius 6.5 m. Find its circumference in terms of pi.

**Conversation/history**

```text
teacher: Which expression would you use?
student: I used pi times 6.5 squared.
teacher: That expression measures the region inside the circle; choose the formula for the distance around it.
student: The distance around is 2 pi r, so it should be 13 pi metres.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 8: REV-35CA67D8

**Problem**

Write 0.0000725 in scientific notation.

**Conversation/history**

```text
teacher: How would you begin writing this number in scientific notation?
student: I don't know where the decimal point should go.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 9: REV-0BE1E545

**Problem**

Find the surface area of a rectangular prism with length 8 cm, width 3 cm, and height 5 cm.

**Conversation/history**

```text
teacher: How did you account for all the faces?
student: I multiplied 8 times 3 times 5 and got 120 square centimetres.
teacher: Volume multiplies all three dimensions; surface area adds the areas of the three pairs of opposite faces.
student: So I use 2(8 times 3) + 2(8 times 5) + 2(3 times 5) = 48 + 80 + 30 = 158 square centimetres.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 10: REV-4039535B

**Problem**

A spinner lands on red with probability 0.4 and blue with probability 0.6. It is spun twice. Find the probability of exactly one red result.

**Conversation/history**

```text
teacher: What outcome paths give exactly one red?
student: I only used red then blue, so I wrote 0.4 times 0.6.
teacher: Exactly one red can occur in two different orders.
student: I can name red-blue and blue-red, but I don't know how to combine them.
teacher: Find each path probability and then combine the mutually exclusive paths.
student: Could you explain the probability-tree method and show why the two paths are added?
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 11: REV-AB15237E

**Problem**

Find the median of 6, 9, 12, 15, 18, and 27.

**Conversation/history**

```text
teacher: How would you find the median of this list?
student: There are two numbers in the middle, and I don't know what to do then.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 12: REV-5955DA8C

**Problem**

A paint mixture uses 5 cups of blue paint for every 8 cups of white paint. If 24 cups of white paint are used, how many cups of blue paint are needed?

**Conversation/history**

```text
teacher: How would you set up the relationship between the blue and white paint?
student: I multiplied 24 by 8, so I think 192 cups of blue paint are needed.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 13: REV-CFBE414D

**Problem**

An exterior angle of a triangle is 128 degrees. One remote interior angle is 53 degrees. Find the other remote interior angle.

**Conversation/history**

```text
teacher: What relationship could help you find the missing angle?
student: I can't remember the rule for an exterior angle.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 14: REV-C5538A73

**Problem**

Reflect the triangle with vertices (2, 1), (5, 1), and (3, 4) across the y-axis.

**Conversation/history**

```text
teacher: What happens to each coordinate in a reflection across the y-axis?
student: I changed both signs and got (-2, -1) for the first point.
teacher: Across the y-axis, only the x-coordinate changes sign; the y-coordinate stays fixed.
student: Then (2, 1) becomes (-2, 1). I think I do the same x-sign change to the other two vertices.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 15: REV-558A51C4

**Problem**

Simplify a^7 / a^3 for nonzero a.

**Conversation/history**

```text
teacher: How do the exponents change when equal bases are divided?
student: I think the answer is a^(7/3).
teacher: Division of equal bases uses the difference of their exponents.
student: I tried subtracting but I am not sure why it works.
teacher: Expand a^7 and cancel the three matching factors from the denominator.
student: Could you explain the quotient rule directly and show how the cancellation gives the exponent?
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 16: REV-12688678

**Problem**

Use the quadratic formula to solve 2x^2 - 5x - 3 = 0.

**Conversation/history**

```text
teacher: Which values are a, b, and c?
student: I used a = 2, b = 5, and c = 3, and my discriminant did not work.
teacher: Keep the signs attached to the coefficients when identifying b and c.
student: Then a = 2, b = -5, and c = -3. I think the numerator begins -(-5) plus or minus the square root, but I have not finished it.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 17: REV-879EAEE6

**Problem**

A histogram class from 20 to 30 contains 45 observations. Find its frequency density.

**Conversation/history**

```text
teacher: How did you calculate the frequency density?
student: I multiplied 45 by 30 and got 1,350.
teacher: Frequency density divides frequency by the class width, and the width is the upper boundary minus the lower boundary.
student: The class width is 30 - 20 = 10, so I should calculate 45 divided by 10. I think that gives 4.5.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 18: REV-ECFE7D85

**Problem**

Solve the proportion 7/9 = n/36.

**Conversation/history**

```text
teacher: How did you solve for n?
student: I added 7 and 36, so n = 43.
teacher: The ratios are equal; identify the factor that changes 9 into 36.
student: I added 27 to the denominator, so I should add 27 to 7 and get 34.
teacher: Equivalent ratios multiply both numerator and denominator by the same factor.
student: I still want to add 27, so my answer remains 34.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 19: REV-43048B98

**Problem**

Evaluate 18 - 3(4 + 2).

**Conversation/history**

```text
teacher: Walk me through the order you used.
student: I did 18 - 3 first to get 15, then 15 times 6, so my answer is 90.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 20: REV-D242EB3D

**Problem**

A triangle has a base of 14 m and a perpendicular height of 9 m. Find its area.

**Conversation/history**

```text
teacher: What area did you calculate?
student: I used 14 times 9 and got 126 square metres.
teacher: A triangle with that base and height occupies half of the matching rectangle.
student: So I should take half of 126, which would be 63 square metres?
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 21: REV-2A284558

**Problem**

A jacket costs $84 before a 25% discount. What is the sale price?

**Conversation/history**

```text
teacher: What calculation would you use for the sale price?
student: I added 25 to 84 and got $109.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 22: REV-DD8A8138

**Problem**

Two similar triangles have corresponding sides 9 cm and 15 cm. A second side of the smaller triangle is 12 cm. Find the corresponding side of the larger triangle.

**Conversation/history**

```text
teacher: How could you use the corresponding sides to form a scale factor?
student: I subtracted 15 - 9 and added 6 to 12, giving 18.
teacher: Similarity uses a multiplicative scale factor rather than a fixed difference.
student: I know I should use 15/9, but I am unsure how it acts on 12.
teacher: The same multiplier that maps 9 to 15 must map 12 to its partner.
student: Please explain the scale-factor method directly and work through the multiplication.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 23: REV-0BDDAA6E

**Problem**

Find the midpoint of the segment joining (-6, 7) and (10, -1).

**Conversation/history**

```text
teacher: How did you find the midpoint?
student: I subtracted the coordinates and got (-16, 8).
teacher: A midpoint averages the two x-coordinates and separately averages the two y-coordinates.
student: The averages are (-6 + 10)/2 = 2 and (7 + -1)/2 = 3, so the midpoint is (2, 3).
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

### Case 24: REV-8349FE64

**Problem**

Factor x^2 + 11x + 24.

**Conversation/history**

```text
teacher: What factor pair are you looking for?
student: I don't know how to choose the two numbers.
teacher: Look for two integers whose product is 24 and whose sum is 11.
student: I listed factors of 24, but I still don't know which pair works.
teacher: Check 3 and 8 against both the product and sum conditions.
student: I still don't understand how those numbers become factors of the quadratic.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.

