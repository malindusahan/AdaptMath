# Blinded Pedagogical-Move Expert Review — Extension 2

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

Frozen extension state-set SHA-256: `0d3834b6aaaee16606e6bbd7c98f2fa3771cad0f5bfc43ccde833f5b7ae9376b`. This packet contains 32 new model-disagreement cases.

## Review cases

### Case 1: R2-E0EB836FD4

**Problem**

Convert 2.75 hours to hours and minutes.

**Conversation/history**

```text
teacher: How did you first interpret 0.75 of an hour?
student: I called it 75 minutes.
teacher: Multiply the fractional hour by 60 minutes.
student: 0.75 times 60 is 45, so the time is 2 hours 45 minutes.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 2: R2-E81CFD5BF5

**Problem**

Solve 5x + 2 = 27.

**Conversation/history**

```text
teacher: What value do you get for x?
student: I divided 27 by 5 first and then subtracted 2, so x is 3.4.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 3: R2-40870CF140

**Problem**

Find the mean of 5, 8, 12, and 15.

**Conversation/history**

```text
teacher: Show your calculation for the mean.
student: The middle numbers are 8 and 12, so the mean is 10.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 4: R2-6ADC8D73C8

**Problem**

Factor x^2 - x - 12.

**Conversation/history**

```text
teacher: Which signs did you first choose?
student: I chose 3 and 4, both positive.
teacher: The product is negative, so the factor numbers need opposite signs and must sum to -1.
student: I think -4 and 3 work because their product is -12 and sum is -1; I have not written the brackets.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 5: R2-027F37664B

**Problem**

Simplify r^11 / r^6 for nonzero r.

**Conversation/history**

```text
teacher: What did you first do with the exponents?
student: I divided 11 by 6.
teacher: A quotient of equal bases subtracts exponents.
student: The new exponent is 11 - 6 = 5, so I think the expression is r^5, though I have not checked by cancellation.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 6: R2-98B6F53F44

**Problem**

A fair six-sided die is rolled once. Find the probability of not rolling a 6.

**Conversation/history**

```text
teacher: What probability did you find?
student: There is one number 6, so the probability is 1/6.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 7: R2-2DCBD96B5C

**Problem**

Find the area of a circle with diameter 16 cm in terms of pi.

**Conversation/history**

```text
teacher: What value did you first use as the radius?
student: I used 16 as the radius.
teacher: The radius is half the diameter before applying A = pi r^2.
student: The radius is 8, so the area setup is pi times 8 squared; I have not simplified it yet.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 8: R2-AD2F9F2E05

**Problem**

A $480 television is discounted by 15%. Find the sale price.

**Conversation/history**

```text
teacher: What did you initially subtract?
student: I subtracted 15 dollars.
teacher: Find 15 percent of 480 as the discount amount.
student: The discount is 0.15 times 480 = 72, so I should subtract 72 from 480 next.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 9: R2-8C92150E74

**Problem**

Solve 2x + 3 > 15.

**Conversation/history**

```text
teacher: What did you first do?
student: I divided 15 by 2 before subtracting 3.
teacher: Subtract 3 from both sides before dividing by positive 2.
student: That gives 2x > 12 and then x > 6. I think the sign stays the same because I divided by a positive number.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 10: R2-47D723EA51

**Problem**

Expand and simplify 4(3x - 2) + x.

**Conversation/history**

```text
teacher: What happened in your first expansion?
student: I wrote 12x - 2 + x.
teacher: Distribute 4 to both terms before collecting like terms.
student: That gives 12x - 8 + x, so the simplified expression is 13x - 8.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 11: R2-309DC81BA2

**Problem**

Find the midpoint of (-3, 8) and (9, 2).

**Conversation/history**

```text
teacher: How did you first combine the points?
student: I added coordinates without dividing and got (6, 10).
teacher: After adding corresponding coordinates, divide each total by 2.
student: The midpoint coordinates are 6/2 and 10/2, so I think the point is (3, 5).
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 12: R2-A48B9320EA

**Problem**

Find the midpoint between (2, -5) and (8, 7).

**Conversation/history**

```text
teacher: What did you initially do with the coordinates?
student: I subtracted the points and got (6, 12).
teacher: Average the x-coordinates and average the y-coordinates separately.
student: The averages are 5 and 1, so the midpoint is (5, 1).
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 13: R2-379126CF6D

**Problem**

The temperature rises from -4 degrees Celsius to 7 degrees Celsius. By how many degrees did it rise?

**Conversation/history**

```text
teacher: What change in temperature do you obtain?
student: I subtracted 4 from 7, so the rise is 3 degrees.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 14: R2-67B18EBE52

**Problem**

Calculate 3/4 - 2/9.

**Conversation/history**

```text
teacher: How did you first subtract the fractions?
student: I subtracted top and bottom to get 1/-5.
teacher: Use a common denominator of 36 before subtracting numerators.
student: Three quarters is 27/36 and two ninths is 8/36, so the result is 19/36.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 15: R2-A71587ACAB

**Problem**

For f(x) = 2x^2 - 1, find f(3).

**Conversation/history**

```text
teacher: What did you calculate first?
student: I used 2 times 3 and then squared the result.
teacher: Substitute 3 for x inside x squared before multiplying by 2.
student: So I use 2(3^2) - 1, which is 18 - 1.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 16: R2-BAED748C21

**Problem**

Translate the point (5, -3) by the vector (-2, 7).

**Conversation/history**

```text
teacher: How did you first use the vector?
student: I changed both coordinate signs.
teacher: Add each vector component to its matching coordinate.
student: The image is (5 - 2, -3 + 7) = (3, 4).
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 17: R2-9705594A70

**Problem**

Solve 3x - 8 < 13.

**Conversation/history**

```text
teacher: What inequality did you get at first?
student: I divided 13 by 3 and then subtracted 8.
teacher: Undo the subtraction of 8 before dividing by 3.
student: Adding 8 gives 3x < 21, and then I divide both sides by 3.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 18: R2-A0E6123F4D

**Problem**

A sector has radius 12 cm and angle 75 degrees. Find its area in terms of pi.

**Conversation/history**

```text
teacher: What fraction of the circle is the sector?
student: I used 75/100 because 75 looks like a percentage.
teacher: Compare the angle with a full 360-degree turn.
student: The fraction is 75/360, but I am unsure how it combines with the circle area.
teacher: Multiply that fraction by pi times radius squared.
student: Could you explain the sector-area method directly and simplify the result?
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 19: R2-56F427D25F

**Problem**

Calculate (-6)(-7).

**Conversation/history**

```text
teacher: What answer did you get?
student: I wrote -42 because both numbers have minus signs.
teacher: Recall the sign rule when two negative factors are multiplied.
student: Two negatives give a positive product, so the answer should be 42.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 20: R2-AF3DC081D9

**Problem**

Calculate 7/10 + 5/12.

**Conversation/history**

```text
teacher: How did you first combine the fractions?
student: I added across and got 12/22.
teacher: Use a common denominator divisible by both 10 and 12.
student: A common denominator is 60, so the fractions become 42/60 and 25/60; I still need to add and simplify.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 21: R2-315419F3AB

**Problem**

Calculate 4.95 divided by 0.5.

**Conversation/history**

```text
teacher: How did you begin?
student: I removed the decimal from 0.5 and used 4.95 divided by 5.
teacher: Multiply both dividend and divisor by 10 to keep the quotient equivalent.
student: That gives 49.5 divided by 5, so I can continue from there.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 22: R2-1C608DF43C

**Problem**

Find the equation of a line with gradient 3 and y-intercept -5.

**Conversation/history**

```text
teacher: What equation did you first write?
student: I wrote y = -5x + 3.
teacher: In y = mx + c, m is the gradient and c is the y-intercept.
student: Then m = 3 and c = -5, so the equation starts y = 3x - 5; I think that is complete.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 23: R2-FD4F5BF1E4

**Problem**

Simplify p^9 / p^4 for nonzero p.

**Conversation/history**

```text
teacher: What did you first do with the exponents?
student: I divided 9 by 4.
teacher: For equal bases in a quotient, subtract the denominator exponent from the numerator exponent.
student: Then the exponent is 9 - 4 = 5, so the result is p^5.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 24: R2-3F7D289E72

**Problem**

Expand (x - 3)(x + 7).

**Conversation/history**

```text
teacher: What did your first expansion include?
student: I only multiplied x by x and -3 by 7.
teacher: Include the two cross-products as well.
student: I now have x^2 + 7x - 3x - 21. I think the middle terms combine, but I have not done it.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 25: R2-5FD2AE40AA

**Problem**

Convert 4.2 metres to centimetres.

**Conversation/history**

```text
teacher: What did you first multiply by?
student: I multiplied by 10 and got 42.
teacher: One metre contains 100 centimetres.
student: Then I should multiply 4.2 by 100. That seems to give 420, but I want to check the unit.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 26: R2-1DB1F4E7DE

**Problem**

Solve 8x - 11 = 53.

**Conversation/history**

```text
teacher: What did you do first?
student: I divided 53 by 8 and then subtracted 11.
teacher: Undo minus 11 before dividing by 8.
student: Adding 11 gives 8x = 64. I know division is next, but I have not written x yet.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 27: R2-7C699FC2EA

**Problem**

Find two thirds of 45.

**Conversation/history**

```text
teacher: How would you calculate the fraction of the amount?
student: I divided 45 by 2 and then multiplied by 3, giving 67.5.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 28: R2-9BCAEC41C2

**Problem**

Factor x^2 + 9x + 20.

**Conversation/history**

```text
teacher: Which numbers did you first try?
student: I used 2 and 10 because they multiply to 20.
teacher: The pair must multiply to 20 and also add to the middle coefficient 9.
student: Four and five meet both conditions, so the brackets should use 4 and 5.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 29: R2-B4F370C3E4

**Problem**

Solve 7x + 5 = 47.

**Conversation/history**

```text
teacher: What did you do first?
student: I divided 47 by 7 before removing 5.
teacher: Undo the addition of 5 first, then divide by 7.
student: Subtracting 5 gives 7x = 42, and dividing by 7 gives x = 6.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 30: R2-5554792E4F

**Problem**

Solve the inequality 4x + 1 > 17.

**Conversation/history**

```text
teacher: What did you initially do?
student: I divided 17 by 4 before removing 1.
teacher: Subtract 1 from both sides before dividing by positive 4.
student: That gives 4x > 16, then x > 4; the direction stays the same because 4 is positive.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 31: R2-E5C43F3D85

**Problem**

For y = 3x - 2, find y when x = 4.

**Conversation/history**

```text
teacher: What value of y do you get?
student: I used 3 + 4 - 2, so y equals 5.
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
### Case 32: R2-73AF762391

**Problem**

Factor x^2 + 13x + 36.

**Conversation/history**

```text
teacher: Which pair did you first choose?
student: I chose 6 and 6 because they multiply to 36.
teacher: The pair must also add to 13.
student: Four and nine multiply to 36 and add to 13, so the factors are (x + 4)(x + 9).
```

Record one move, confidence 1-3, and an optional one-sentence reason in the CSV form.
