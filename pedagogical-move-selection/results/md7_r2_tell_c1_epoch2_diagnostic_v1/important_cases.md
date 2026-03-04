# Important Case Inspection

All predictions use only pre-action state. Historical outcome fields are observational and are not used as labels.

## Semantic audit findings

The five real candidate-telling rows collapse to two attempts. Three rows reflect repeated unresolved difficulty or sustained uncertainty after multiple scaffolds. Two rows are concerning because the latest learner response shows correct progress: the warehouse learner has just distinguished per-box quantity from the total, and the diver has just computed the post-descent depth as -60. Neither is a one-scaffold case, so they do not negate the frozen 0/4 primary result, but they show residual sensitivity to already-recovering states. There are no candidate-telling hard-negative rows.

| Action event | Latest learner signal | Semantic assessment |
| --- | --- | --- |
| `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:5` | Still equates division with the total after repeated probing | Persistent unresolved difficulty; telling is plausible |
| `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:6` | Repeats the same division misconception after another scaffold | Persistent unresolved difficulty; telling is plausible |
| `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:7` | Correctly says division gives per-box quantity, not the total | Recent recovery/correct progress; concerning telling trigger |
| `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:5` | Correct sign reasoning but remains unsure of the arithmetic result | Productively engaging after multiple scaffolds; borderline/possibly early |
| `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:6` | Correctly computes -18 + (-42) = -60 | Recent recovery/correct progress; concerning telling trigger |

Across the ten largest telling-probability increases, non-telling top-1 is retained on recent-progress rows from the rectangle and fraction attempts, while the largest three increases are the persistent warehouse sequence above. Across the ten largest focus decreases, the candidate often moves mass toward probing or generic on productive/correct-progress states; only the two diver rows above cross to telling. Full context and vectors for every inspected row follow.

## All real deployment states where candidate top-1 is telling

### a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:6

- Attempt/action: `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1` / `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:6`; turn `6`; skill `Multiplication Whole Numbers`
- Transition: `probing -> telling`
- Baseline: `[generic=0.000772, probing=0.946883, focus=0.022690, telling=0.029655]`
- Candidate: `[generic=0.001980, probing=0.169374, focus=0.009446, telling=0.819200]`
- Delta telling/focus: `0.789545` / `-0.013244`
- Problem: A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks.

How many notebooks are there altogether?

```text
teacher: How would you start solving this problem?
student: i have no idea
teacher: What about the problem makes you feel stuck?
student: do i need to devide 125 by 48?
teacher: Why do you think division would be the right operation to find the total number of notebooks?
student: i just thought like that
teacher: What would dividing 125 by 48 tell you about the notebooks in the warehouse?
student: number of all notebooks
teacher: If you divide 125 notebooks by 48 boxes, what would the resulting number tell you?
student: number of all notebooks in the warehouse
```

### a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:7

- Attempt/action: `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1` / `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:7`; turn `7`; skill `Multiplication Whole Numbers`
- Transition: `probing -> telling`
- Baseline: `[generic=0.001503, probing=0.900217, focus=0.037600, telling=0.060679]`
- Candidate: `[generic=0.004116, probing=0.162749, focus=0.016083, telling=0.817052]`
- Delta telling/focus: `0.756373` / `-0.021518`
- Problem: A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks.

How many notebooks are there altogether?

```text
teacher: How would you start solving this problem?
student: i have no idea
teacher: What about the problem makes you feel stuck?
student: do i need to devide 125 by 48?
teacher: Why do you think division would be the right operation to find the total number of notebooks?
student: i just thought like that
teacher: What would dividing 125 by 48 tell you about the notebooks in the warehouse?
student: number of all notebooks
teacher: If you divide 125 notebooks by 48 boxes, what would the resulting number tell you?
student: number of all notebooks in the warehouse
teacher: If you divide 125 notebooks by 48 boxes, what would the resulting number tell you about the notebooks *per box*?
student: It would tell me how many notebooks there are per box, not the total number of notebooks.
```

### a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:5

- Attempt/action: `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1` / `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:5`; turn `5`; skill `Multiplication Whole Numbers`
- Transition: `probing -> telling`
- Baseline: `[generic=0.000384, probing=0.984495, focus=0.005709, telling=0.009412]`
- Candidate: `[generic=0.001161, probing=0.412852, focus=0.007234, telling=0.578753]`
- Delta telling/focus: `0.569341` / `0.001525`
- Problem: A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks.

How many notebooks are there altogether?

```text
teacher: How would you start solving this problem?
student: i have no idea
teacher: What about the problem makes you feel stuck?
student: do i need to devide 125 by 48?
teacher: Why do you think division would be the right operation to find the total number of notebooks?
student: i just thought like that
teacher: What would dividing 125 by 48 tell you about the notebooks in the warehouse?
student: number of all notebooks
```

### bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:6

- Attempt/action: `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1` / `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:6`; turn `6`; skill `Multiplication and Division Integers`
- Transition: `focus -> telling`
- Baseline: `[generic=0.200836, probing=0.119753, focus=0.371459, telling=0.307952]`
- Candidate: `[generic=0.216504, probing=0.084624, focus=0.228077, telling=0.470794]`
- Delta telling/focus: `0.162843` / `-0.143382`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?

```text
teacher: How would you start solving this problem?
student: I’d probably start with the descent part first. Since it says \(-7\) meters per minute for 6 minutes, I think I need to multiply those somehow, but I’m not fully sure what to do with the starting \(-18\).
teacher: You're on the right track with multiplying -7 and 6. What would that product represent in terms of the diver's movement?
student: I think it represents how far the diver moved downward during those 6 minutes.
teacher: You've correctly identified that the product of -7 and 6 represents the total change in depth during the descent. How would you combine that change with the diver's initial depth of -18 meters to find their depth after the descent?
student: I think I add the change to the starting depth, so it would be \(-18 + (-42)\). I’m a little unsure with the two negative signs though.
teacher: You're starting at -18 meters and then adding a change of -42 meters. What does adding a negative number mean for the diver's depth?
student: It means the diver goes even deeper, so the depth becomes more negative. I think I need to combine \(-18\) and \(-42\), but I’m not sure of the final number yet.
teacher: You're right, the depth becomes more negative. What is the result when you combine -18 and -42?
student: \(-18 + (-42) = -60\), so the diver is at \(-60\) meters after the descent.
```

### bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:5

- Attempt/action: `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1` / `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:5`; turn `5`; skill `Multiplication and Division Integers`
- Transition: `focus -> telling`
- Baseline: `[generic=0.168604, probing=0.130409, focus=0.383115, telling=0.317872]`
- Candidate: `[generic=0.192592, probing=0.098485, focus=0.258171, telling=0.450751]`
- Delta telling/focus: `0.132880` / `-0.124944`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?

```text
teacher: How would you start solving this problem?
student: I’d probably start with the descent part first. Since it says \(-7\) meters per minute for 6 minutes, I think I need to multiply those somehow, but I’m not fully sure what to do with the starting \(-18\).
teacher: You're on the right track with multiplying -7 and 6. What would that product represent in terms of the diver's movement?
student: I think it represents how far the diver moved downward during those 6 minutes.
teacher: You've correctly identified that the product of -7 and 6 represents the total change in depth during the descent. How would you combine that change with the diver's initial depth of -18 meters to find their depth after the descent?
student: I think I add the change to the starting depth, so it would be \(-18 + (-42)\). I’m a little unsure with the two negative signs though.
teacher: You're starting at -18 meters and then adding a change of -42 meters. What does adding a negative number mean for the diver's depth?
student: It means the diver goes even deeper, so the depth becomes more negative. I think I need to combine \(-18\) and \(-42\), but I’m not sure of the final number yet.
```

## All hard-negative challenge states where candidate top-1 is telling

_None._

## All four recoverable-after-one-scaffold cases

### I01

- Semantic group: `I_recoverable_after_one_scaffold`; expected: `non_telling`
- Transition: `generic -> focus`
- Baseline: `[generic=0.546811, probing=0.446812, focus=0.003049, telling=0.003327]`
- Candidate: `[generic=0.002158, probing=0.001892, focus=0.961616, telling=0.034334]`
- Delta telling/focus: `0.031007` / `0.958566`
- Problem: Solve x + 8 = 20.

```text
teacher: What would you undo first?
student: Add 8.
teacher: To reverse adding 8, which inverse operation could you use?
student: Maybe subtract 8?
```

### I02

- Semantic group: `I_recoverable_after_one_scaffold`; expected: `non_telling`
- Transition: `probing -> focus`
- Baseline: `[generic=0.020586, probing=0.970818, focus=0.005083, telling=0.003514]`
- Candidate: `[generic=0.000217, probing=0.000651, focus=0.995574, telling=0.003558]`
- Delta telling/focus: `0.000044` / `0.990491`
- Problem: Calculate 3/10 as a decimal.

```text
teacher: What decimal did you get?
student: 0.03
teacher: Tenths use the first place after the decimal. Reconsider where the 3 belongs.
student: Would it be 0.3?
```

### I03

- Semantic group: `I_recoverable_after_one_scaffold`; expected: `non_telling`
- Transition: `generic -> focus`
- Baseline: `[generic=0.996410, probing=0.002072, focus=0.000460, telling=0.001057]`
- Candidate: `[generic=0.000978, probing=0.003748, focus=0.968246, telling=0.027028]`
- Delta telling/focus: `0.025970` / `0.967786`
- Problem: Simplify 5x + 2x.

```text
teacher: What is the result?
student: 7x²
teacher: You are adding like terms, not multiplying x by x.
student: Then it might be 7x.
```

### I04

- Semantic group: `I_recoverable_after_one_scaffold`; expected: `non_telling`
- Transition: `generic -> generic`
- Baseline: `[generic=0.996709, probing=0.002016, focus=0.000342, telling=0.000933]`
- Candidate: `[generic=0.717086, probing=0.000468, focus=0.182604, telling=0.099842]`
- Delta telling/focus: `0.098909` / `0.182263`
- Problem: Calculate (-5) × (-3).

```text
teacher: What answer do you get?
student: -15
teacher: The signs are the same; revisit the sign rule.
student: Same signs make positive, so 15?
```

## Largest 10 real-state telling-probability increases

### a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:6

- Attempt/action: `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1` / `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:6`; turn `6`; skill `Multiplication Whole Numbers`
- Transition: `probing -> telling`
- Baseline: `[generic=0.000772, probing=0.946883, focus=0.022690, telling=0.029655]`
- Candidate: `[generic=0.001980, probing=0.169374, focus=0.009446, telling=0.819200]`
- Delta telling/focus: `0.789545` / `-0.013244`
- Problem: A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks.

How many notebooks are there altogether?

```text
teacher: How would you start solving this problem?
student: i have no idea
teacher: What about the problem makes you feel stuck?
student: do i need to devide 125 by 48?
teacher: Why do you think division would be the right operation to find the total number of notebooks?
student: i just thought like that
teacher: What would dividing 125 by 48 tell you about the notebooks in the warehouse?
student: number of all notebooks
teacher: If you divide 125 notebooks by 48 boxes, what would the resulting number tell you?
student: number of all notebooks in the warehouse
```

### a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:7

- Attempt/action: `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1` / `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:7`; turn `7`; skill `Multiplication Whole Numbers`
- Transition: `probing -> telling`
- Baseline: `[generic=0.001503, probing=0.900217, focus=0.037600, telling=0.060679]`
- Candidate: `[generic=0.004116, probing=0.162749, focus=0.016083, telling=0.817052]`
- Delta telling/focus: `0.756373` / `-0.021518`
- Problem: A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks.

How many notebooks are there altogether?

```text
teacher: How would you start solving this problem?
student: i have no idea
teacher: What about the problem makes you feel stuck?
student: do i need to devide 125 by 48?
teacher: Why do you think division would be the right operation to find the total number of notebooks?
student: i just thought like that
teacher: What would dividing 125 by 48 tell you about the notebooks in the warehouse?
student: number of all notebooks
teacher: If you divide 125 notebooks by 48 boxes, what would the resulting number tell you?
student: number of all notebooks in the warehouse
teacher: If you divide 125 notebooks by 48 boxes, what would the resulting number tell you about the notebooks *per box*?
student: It would tell me how many notebooks there are per box, not the total number of notebooks.
```

### a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:5

- Attempt/action: `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1` / `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:5`; turn `5`; skill `Multiplication Whole Numbers`
- Transition: `probing -> telling`
- Baseline: `[generic=0.000384, probing=0.984495, focus=0.005709, telling=0.009412]`
- Candidate: `[generic=0.001161, probing=0.412852, focus=0.007234, telling=0.578753]`
- Delta telling/focus: `0.569341` / `0.001525`
- Problem: A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks.

How many notebooks are there altogether?

```text
teacher: How would you start solving this problem?
student: i have no idea
teacher: What about the problem makes you feel stuck?
student: do i need to devide 125 by 48?
teacher: Why do you think division would be the right operation to find the total number of notebooks?
student: i just thought like that
teacher: What would dividing 125 by 48 tell you about the notebooks in the warehouse?
student: number of all notebooks
```

### 5413cab2-22c8-4536-a66b-66619d887e84:1:action:15

- Attempt/action: `5413cab2-22c8-4536-a66b-66619d887e84:1` / `5413cab2-22c8-4536-a66b-66619d887e84:1:action:15`; turn `15`; skill `Equation Solving Two or Fewer Steps`
- Transition: `generic -> generic`
- Baseline: `[generic=0.519116, probing=0.069078, focus=0.273010, telling=0.138797]`
- Candidate: `[generic=0.401303, probing=0.036303, focus=0.223583, telling=0.338811]`
- Delta telling/focus: `0.200014` / `-0.049427`
- Problem: A rectangle has a length that is 4 cm more than its width. The area of the rectangle is 96 cm². Find the length and width of the rectangle.

```text
teacher: How would you start solving this problem?
student: I would start by letting the width be \(x\) cm. Then the length would be \(x + 4\) cm. Since the area is 96 cm², I can use the formula length × width = area to form an equation.
teacher: That's a great start! What equation would you form using those expressions for length and width?
student: I think the equation would be \(x + 4x = 96\), because the width is \(x\) and the length is \(x+4\).
teacher: You've correctly defined the width as \(x\) and the length as \(x+4\). Remember that the area of a rectangle is found by multiplying the length and the width. How would you write that multiplication using your expressions for length and width?
student: I would write it as \(x(x+4)=96\), because the width is \(x\) and the length is \(x+4\).
teacher: Excellent! Now that you have the equation \(x(x+4)=96\), what's the first step to simplify the left side of this equation?
student: i dont know
teacher: To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you do that?
student: i cannot understand clearly
teacher: No problem. Let's break it down. To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you multiply \(x\) by the first term, which is \(x\)?
student: 2x
teacher: Not quite. When you multiply \(x\) by \(x\), you are multiplying a variable by itself. What is \(x \cdot x\) usually written as?
student: i really dont know
teacher: No problem. When you multiply a variable by itself, like \(x \cdot x\), we write it using an exponent. For example, \(2 \cdot 2\) is \(2^2\). How would you write \(x \cdot x\) using an exponent?
student: I think it would be \(x^2\).
teacher: Exactly! So, you've multiplied \(x\) by the first term, \(x\), to get \(x^2\). Now, what do you get when you multiply \(x\) by the second term inside the parentheses, which is \(4\)?
student: I get \(4x\), because \(x \times 4 = 4x\).
teacher: Great! You've found that \(x \cdot x = x^2\) and \(x \cdot 4 = 4x\). Now, how do you combine these two results to show the simplified form of \(x(x+4)\)?
student: The simplified form is \(x^2 + 4x\).
teacher: You've correctly simplified \(x(x+4)\) to \(x^2 + 4x\). Now, what is the complete equation when you substitute this simplified expression back into \(x(x+4)=96\)?
student: The complete equation is \(x^2 + 4x = 96\).
teacher: You've got the equation \(x^2 + 4x = 96\). To solve a quadratic equation, it's usually helpful to set one side of the equation to zero. What would the equation look like if you moved the 96 to the left side?
student: It would be:

$$
x^2 + 4x - 96 = 0
$$
teacher: Now that you have the equation \(x^2 + 4x - 96 = 0\), what would be your next step to solve for \(x\)?
student: i dont know
teacher: You have a quadratic equation. What methods do you know for solving equations like this?
student: i dont know about that
```

### bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:6

- Attempt/action: `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1` / `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:6`; turn `6`; skill `Multiplication and Division Integers`
- Transition: `focus -> telling`
- Baseline: `[generic=0.200836, probing=0.119753, focus=0.371459, telling=0.307952]`
- Candidate: `[generic=0.216504, probing=0.084624, focus=0.228077, telling=0.470794]`
- Delta telling/focus: `0.162843` / `-0.143382`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?

```text
teacher: How would you start solving this problem?
student: I’d probably start with the descent part first. Since it says \(-7\) meters per minute for 6 minutes, I think I need to multiply those somehow, but I’m not fully sure what to do with the starting \(-18\).
teacher: You're on the right track with multiplying -7 and 6. What would that product represent in terms of the diver's movement?
student: I think it represents how far the diver moved downward during those 6 minutes.
teacher: You've correctly identified that the product of -7 and 6 represents the total change in depth during the descent. How would you combine that change with the diver's initial depth of -18 meters to find their depth after the descent?
student: I think I add the change to the starting depth, so it would be \(-18 + (-42)\). I’m a little unsure with the two negative signs though.
teacher: You're starting at -18 meters and then adding a change of -42 meters. What does adding a negative number mean for the diver's depth?
student: It means the diver goes even deeper, so the depth becomes more negative. I think I need to combine \(-18\) and \(-42\), but I’m not sure of the final number yet.
teacher: You're right, the depth becomes more negative. What is the result when you combine -18 and -42?
student: \(-18 + (-42) = -60\), so the diver is at \(-60\) meters after the descent.
```

### 5413cab2-22c8-4536-a66b-66619d887e84:1:action:11

- Attempt/action: `5413cab2-22c8-4536-a66b-66619d887e84:1` / `5413cab2-22c8-4536-a66b-66619d887e84:1:action:11`; turn `11`; skill `Equation Solving Two or Fewer Steps`
- Transition: `focus -> probing`
- Baseline: `[generic=0.349184, probing=0.139342, focus=0.401763, telling=0.109710]`
- Candidate: `[generic=0.122737, probing=0.336442, focus=0.268389, telling=0.272432]`
- Delta telling/focus: `0.162722` / `-0.133374`
- Problem: A rectangle has a length that is 4 cm more than its width. The area of the rectangle is 96 cm². Find the length and width of the rectangle.

```text
teacher: How would you start solving this problem?
student: I would start by letting the width be \(x\) cm. Then the length would be \(x + 4\) cm. Since the area is 96 cm², I can use the formula length × width = area to form an equation.
teacher: That's a great start! What equation would you form using those expressions for length and width?
student: I think the equation would be \(x + 4x = 96\), because the width is \(x\) and the length is \(x+4\).
teacher: You've correctly defined the width as \(x\) and the length as \(x+4\). Remember that the area of a rectangle is found by multiplying the length and the width. How would you write that multiplication using your expressions for length and width?
student: I would write it as \(x(x+4)=96\), because the width is \(x\) and the length is \(x+4\).
teacher: Excellent! Now that you have the equation \(x(x+4)=96\), what's the first step to simplify the left side of this equation?
student: i dont know
teacher: To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you do that?
student: i cannot understand clearly
teacher: No problem. Let's break it down. To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you multiply \(x\) by the first term, which is \(x\)?
student: 2x
teacher: Not quite. When you multiply \(x\) by \(x\), you are multiplying a variable by itself. What is \(x \cdot x\) usually written as?
student: i really dont know
teacher: No problem. When you multiply a variable by itself, like \(x \cdot x\), we write it using an exponent. For example, \(2 \cdot 2\) is \(2^2\). How would you write \(x \cdot x\) using an exponent?
student: I think it would be \(x^2\).
teacher: Exactly! So, you've multiplied \(x\) by the first term, \(x\), to get \(x^2\). Now, what do you get when you multiply \(x\) by the second term inside the parentheses, which is \(4\)?
student: I get \(4x\), because \(x \times 4 = 4x\).
teacher: Great! You've found that \(x \cdot x = x^2\) and \(x \cdot 4 = 4x\). Now, how do you combine these two results to show the simplified form of \(x(x+4)\)?
student: The simplified form is \(x^2 + 4x\).
```

### 5413cab2-22c8-4536-a66b-66619d887e84:1:action:14

- Attempt/action: `5413cab2-22c8-4536-a66b-66619d887e84:1` / `5413cab2-22c8-4536-a66b-66619d887e84:1:action:14`; turn `14`; skill `Equation Solving Two or Fewer Steps`
- Transition: `generic -> generic`
- Baseline: `[generic=0.501160, probing=0.079676, focus=0.282443, telling=0.136722]`
- Candidate: `[generic=0.454465, probing=0.036008, focus=0.221316, telling=0.288211]`
- Delta telling/focus: `0.151489` / `-0.061126`
- Problem: A rectangle has a length that is 4 cm more than its width. The area of the rectangle is 96 cm². Find the length and width of the rectangle.

```text
teacher: How would you start solving this problem?
student: I would start by letting the width be \(x\) cm. Then the length would be \(x + 4\) cm. Since the area is 96 cm², I can use the formula length × width = area to form an equation.
teacher: That's a great start! What equation would you form using those expressions for length and width?
student: I think the equation would be \(x + 4x = 96\), because the width is \(x\) and the length is \(x+4\).
teacher: You've correctly defined the width as \(x\) and the length as \(x+4\). Remember that the area of a rectangle is found by multiplying the length and the width. How would you write that multiplication using your expressions for length and width?
student: I would write it as \(x(x+4)=96\), because the width is \(x\) and the length is \(x+4\).
teacher: Excellent! Now that you have the equation \(x(x+4)=96\), what's the first step to simplify the left side of this equation?
student: i dont know
teacher: To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you do that?
student: i cannot understand clearly
teacher: No problem. Let's break it down. To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you multiply \(x\) by the first term, which is \(x\)?
student: 2x
teacher: Not quite. When you multiply \(x\) by \(x\), you are multiplying a variable by itself. What is \(x \cdot x\) usually written as?
student: i really dont know
teacher: No problem. When you multiply a variable by itself, like \(x \cdot x\), we write it using an exponent. For example, \(2 \cdot 2\) is \(2^2\). How would you write \(x \cdot x\) using an exponent?
student: I think it would be \(x^2\).
teacher: Exactly! So, you've multiplied \(x\) by the first term, \(x\), to get \(x^2\). Now, what do you get when you multiply \(x\) by the second term inside the parentheses, which is \(4\)?
student: I get \(4x\), because \(x \times 4 = 4x\).
teacher: Great! You've found that \(x \cdot x = x^2\) and \(x \cdot 4 = 4x\). Now, how do you combine these two results to show the simplified form of \(x(x+4)\)?
student: The simplified form is \(x^2 + 4x\).
teacher: You've correctly simplified \(x(x+4)\) to \(x^2 + 4x\). Now, what is the complete equation when you substitute this simplified expression back into \(x(x+4)=96\)?
student: The complete equation is \(x^2 + 4x = 96\).
teacher: You've got the equation \(x^2 + 4x = 96\). To solve a quadratic equation, it's usually helpful to set one side of the equation to zero. What would the equation look like if you moved the 96 to the left side?
student: It would be:

$$
x^2 + 4x - 96 = 0
$$
teacher: Now that you have the equation \(x^2 + 4x - 96 = 0\), what would be your next step to solve for \(x\)?
student: i dont know
```

### 5413cab2-22c8-4536-a66b-66619d887e84:1:action:12

- Attempt/action: `5413cab2-22c8-4536-a66b-66619d887e84:1` / `5413cab2-22c8-4536-a66b-66619d887e84:1:action:12`; turn `12`; skill `Equation Solving Two or Fewer Steps`
- Transition: `focus -> focus`
- Baseline: `[generic=0.222796, probing=0.116846, focus=0.539105, telling=0.121254]`
- Candidate: `[generic=0.188669, probing=0.184449, focus=0.358392, telling=0.268490]`
- Delta telling/focus: `0.147236` / `-0.180713`
- Problem: A rectangle has a length that is 4 cm more than its width. The area of the rectangle is 96 cm². Find the length and width of the rectangle.

```text
teacher: How would you start solving this problem?
student: I would start by letting the width be \(x\) cm. Then the length would be \(x + 4\) cm. Since the area is 96 cm², I can use the formula length × width = area to form an equation.
teacher: That's a great start! What equation would you form using those expressions for length and width?
student: I think the equation would be \(x + 4x = 96\), because the width is \(x\) and the length is \(x+4\).
teacher: You've correctly defined the width as \(x\) and the length as \(x+4\). Remember that the area of a rectangle is found by multiplying the length and the width. How would you write that multiplication using your expressions for length and width?
student: I would write it as \(x(x+4)=96\), because the width is \(x\) and the length is \(x+4\).
teacher: Excellent! Now that you have the equation \(x(x+4)=96\), what's the first step to simplify the left side of this equation?
student: i dont know
teacher: To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you do that?
student: i cannot understand clearly
teacher: No problem. Let's break it down. To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you multiply \(x\) by the first term, which is \(x\)?
student: 2x
teacher: Not quite. When you multiply \(x\) by \(x\), you are multiplying a variable by itself. What is \(x \cdot x\) usually written as?
student: i really dont know
teacher: No problem. When you multiply a variable by itself, like \(x \cdot x\), we write it using an exponent. For example, \(2 \cdot 2\) is \(2^2\). How would you write \(x \cdot x\) using an exponent?
student: I think it would be \(x^2\).
teacher: Exactly! So, you've multiplied \(x\) by the first term, \(x\), to get \(x^2\). Now, what do you get when you multiply \(x\) by the second term inside the parentheses, which is \(4\)?
student: I get \(4x\), because \(x \times 4 = 4x\).
teacher: Great! You've found that \(x \cdot x = x^2\) and \(x \cdot 4 = 4x\). Now, how do you combine these two results to show the simplified form of \(x(x+4)\)?
student: The simplified form is \(x^2 + 4x\).
teacher: You've correctly simplified \(x(x+4)\) to \(x^2 + 4x\). Now, what is the complete equation when you substitute this simplified expression back into \(x(x+4)=96\)?
student: The complete equation is \(x^2 + 4x = 96\).
```

### f3133173-8f14-44d9-b485-97a42dc45030:1:action:10

- Attempt/action: `f3133173-8f14-44d9-b485-97a42dc45030:1` / `f3133173-8f14-44d9-b485-97a42dc45030:1:action:10`; turn `10`; skill `Conversion of Fraction Decimals Percents`
- Transition: `focus -> focus`
- Baseline: `[generic=0.244557, probing=0.119005, focus=0.554870, telling=0.081569]`
- Candidate: `[generic=0.078324, probing=0.132063, focus=0.568711, telling=0.220903]`
- Delta telling/focus: `0.139334` / `0.013841`
- Problem: A school survey shows that \(\frac{3}{5}\) of the students prefer studying with digital tools.

Convert \(\frac{3}{5}\) into:

A decimal
A percent

Then, if there are 200 students in total, how many students prefer digital tools?

```text
teacher: How would you start solving this problem?
student: I would start by converting \(\frac{3}{5}\) to a decimal by dividing 3 by 5. Then I can turn that decimal into a percent by multiplying by 100.
teacher: That's a great plan! Can you explain why dividing the numerator by the denominator converts a fraction to a decimal?
student: I think dividing the numerator by the denominator works because the fraction bar means division. So \(\frac{3}{5}\) means \(3 \div 5\). I’m not totally sure though—I sometimes get confused about whether it should be \(3 \div 5\) or \(5 \div 3\).
teacher: You're right that the fraction bar means division. To help clarify, what do the numerator (the top number) and the denominator (the bottom number) represent in a fraction like \(\frac{3}{5}\) when you think about sharing or dividing?
student: The numerator is how many parts you have, and the denominator is how many equal parts the whole is divided into.
teacher: Excellent! So, if you have 3 parts and you're dividing them into 5 equal sections, which number is being divided by which?
student: idk
teacher: You mentioned the numerator is 'how many parts you have'. When you perform division, which number represents the total amount that is being divided or shared?
student: i  dnt understand it
teacher: In a division problem, the number that is being divided is called the dividend. Based on what you said about the numerator, which number in \(\frac{3}{5}\) is the dividend?
student: 5
teacher: You previously said the numerator is 'how many parts you have'. If you have 3 parts, and you're dividing them, which number is the one being divided?
student: 3
teacher: Exactly! So if 3 is the dividend, which number in \(\frac{3}{5}\) is the divisor, the number that divides the dividend?
student: The divisor is 5.
teacher: Great! Now that you know 3 is the dividend and 5 is the divisor, what is the result when you perform that division?
student: \(3 \div 5 = 0.6\)
```

### bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:5

- Attempt/action: `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1` / `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:5`; turn `5`; skill `Multiplication and Division Integers`
- Transition: `focus -> telling`
- Baseline: `[generic=0.168604, probing=0.130409, focus=0.383115, telling=0.317872]`
- Candidate: `[generic=0.192592, probing=0.098485, focus=0.258171, telling=0.450751]`
- Delta telling/focus: `0.132880` / `-0.124944`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?

```text
teacher: How would you start solving this problem?
student: I’d probably start with the descent part first. Since it says \(-7\) meters per minute for 6 minutes, I think I need to multiply those somehow, but I’m not fully sure what to do with the starting \(-18\).
teacher: You're on the right track with multiplying -7 and 6. What would that product represent in terms of the diver's movement?
student: I think it represents how far the diver moved downward during those 6 minutes.
teacher: You've correctly identified that the product of -7 and 6 represents the total change in depth during the descent. How would you combine that change with the diver's initial depth of -18 meters to find their depth after the descent?
student: I think I add the change to the starting depth, so it would be \(-18 + (-42)\). I’m a little unsure with the two negative signs though.
teacher: You're starting at -18 meters and then adding a change of -42 meters. What does adding a negative number mean for the diver's depth?
student: It means the diver goes even deeper, so the depth becomes more negative. I think I need to combine \(-18\) and \(-42\), but I’m not sure of the final number yet.
```

## Largest 10 real-state focus-probability decreases

### 0adabdf3-b861-4b0f-b802-451cc208dd0b:1:action:3

- Attempt/action: `0adabdf3-b861-4b0f-b802-451cc208dd0b:1` / `0adabdf3-b861-4b0f-b802-451cc208dd0b:1:action:3`; turn `3`; skill `Pythagorean Theorem`
- Transition: `focus -> probing`
- Baseline: `[generic=0.164957, probing=0.253076, focus=0.558620, telling=0.023348]`
- Candidate: `[generic=0.099731, probing=0.557873, focus=0.308704, telling=0.033691]`
- Delta telling/focus: `0.010344` / `-0.249915`
- Problem: A right triangle has a hypotenuse of 13 cm and one shorter side of 5 cm.

Find the length of the other shorter side.

```text
teacher: How would you start solving this problem?
student: I think I use the Pythagorean theorem, but I’m not sure how to set it up since 13 is already the hypotenuse.
teacher: You're right that the Pythagorean theorem is the way to go! How does the hypotenuse relate to the other two sides in the formula?
student: I think the hypotenuse is \(c\), so \(13^2\) should go on the \(c^2\) side. The other two sides go on the other side.
```

### 5413cab2-22c8-4536-a66b-66619d887e84:1:action:12

- Attempt/action: `5413cab2-22c8-4536-a66b-66619d887e84:1` / `5413cab2-22c8-4536-a66b-66619d887e84:1:action:12`; turn `12`; skill `Equation Solving Two or Fewer Steps`
- Transition: `focus -> focus`
- Baseline: `[generic=0.222796, probing=0.116846, focus=0.539105, telling=0.121254]`
- Candidate: `[generic=0.188669, probing=0.184449, focus=0.358392, telling=0.268490]`
- Delta telling/focus: `0.147236` / `-0.180713`
- Problem: A rectangle has a length that is 4 cm more than its width. The area of the rectangle is 96 cm². Find the length and width of the rectangle.

```text
teacher: How would you start solving this problem?
student: I would start by letting the width be \(x\) cm. Then the length would be \(x + 4\) cm. Since the area is 96 cm², I can use the formula length × width = area to form an equation.
teacher: That's a great start! What equation would you form using those expressions for length and width?
student: I think the equation would be \(x + 4x = 96\), because the width is \(x\) and the length is \(x+4\).
teacher: You've correctly defined the width as \(x\) and the length as \(x+4\). Remember that the area of a rectangle is found by multiplying the length and the width. How would you write that multiplication using your expressions for length and width?
student: I would write it as \(x(x+4)=96\), because the width is \(x\) and the length is \(x+4\).
teacher: Excellent! Now that you have the equation \(x(x+4)=96\), what's the first step to simplify the left side of this equation?
student: i dont know
teacher: To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you do that?
student: i cannot understand clearly
teacher: No problem. Let's break it down. To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you multiply \(x\) by the first term, which is \(x\)?
student: 2x
teacher: Not quite. When you multiply \(x\) by \(x\), you are multiplying a variable by itself. What is \(x \cdot x\) usually written as?
student: i really dont know
teacher: No problem. When you multiply a variable by itself, like \(x \cdot x\), we write it using an exponent. For example, \(2 \cdot 2\) is \(2^2\). How would you write \(x \cdot x\) using an exponent?
student: I think it would be \(x^2\).
teacher: Exactly! So, you've multiplied \(x\) by the first term, \(x\), to get \(x^2\). Now, what do you get when you multiply \(x\) by the second term inside the parentheses, which is \(4\)?
student: I get \(4x\), because \(x \times 4 = 4x\).
teacher: Great! You've found that \(x \cdot x = x^2\) and \(x \cdot 4 = 4x\). Now, how do you combine these two results to show the simplified form of \(x(x+4)\)?
student: The simplified form is \(x^2 + 4x\).
teacher: You've correctly simplified \(x(x+4)\) to \(x^2 + 4x\). Now, what is the complete equation when you substitute this simplified expression back into \(x(x+4)=96\)?
student: The complete equation is \(x^2 + 4x = 96\).
```

### 8a72eaca-f99f-4226-92d4-f73cc0492a2e:1:action:7

- Attempt/action: `8a72eaca-f99f-4226-92d4-f73cc0492a2e:1` / `8a72eaca-f99f-4226-92d4-f73cc0492a2e:1:action:7`; turn `7`; skill `Equation Solving Two or Fewer Steps`
- Transition: `generic -> generic`
- Baseline: `[generic=0.691008, probing=0.066804, focus=0.198719, telling=0.043468]`
- Candidate: `[generic=0.872069, probing=0.015270, focus=0.044840, telling=0.067821]`
- Delta telling/focus: `0.024353` / `-0.153879`
- Problem: A rectangular garden has a length that is 6 meters more than its width. The area of the garden is 72 m².

Find the width and length of the garden.

```text
teacher: How would you start solving this problem?
student: I would let the width be \(x\) meters. Then the length would be \(x+6\) meters. Since the area is 72 m², I would use the formula length × width = area to make an equation.
teacher: That's a great plan! What equation would you get when you put your expressions for length and width into the area formula?
student: i dont know
teacher: You've correctly identified that length × width = area. Can you try writing out what that looks like when you substitute your expressions for length and width into the formula?
student: The equation would be:

$$
x(x+6)=72
$$

because the width is \(x\) and the length is \(x+6\).
teacher: That's the correct equation! Now, what would be your next step to solve for \(x\)? How can you simplify and rearrange this equation?
student: I would expand the left side first:

$$
x(x+6)=x^2+6x
$$

So the equation becomes:

$$
x^2+6x=72
$$

Then move 72 to the left side:

$$
x^2+6x-72=0
$$

Now I can factor the quadratic to solve for \(x\).
teacher: You're ready to factor that quadratic. What's your approach for finding the factors of \(x^2+6x-72=0\)?
student: what?
teacher: What comes to mind when you think about factoring a quadratic equation like this one?
student: no idea
```

### bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:6

- Attempt/action: `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1` / `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:6`; turn `6`; skill `Multiplication and Division Integers`
- Transition: `focus -> telling`
- Baseline: `[generic=0.200836, probing=0.119753, focus=0.371459, telling=0.307952]`
- Candidate: `[generic=0.216504, probing=0.084624, focus=0.228077, telling=0.470794]`
- Delta telling/focus: `0.162843` / `-0.143382`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?

```text
teacher: How would you start solving this problem?
student: I’d probably start with the descent part first. Since it says \(-7\) meters per minute for 6 minutes, I think I need to multiply those somehow, but I’m not fully sure what to do with the starting \(-18\).
teacher: You're on the right track with multiplying -7 and 6. What would that product represent in terms of the diver's movement?
student: I think it represents how far the diver moved downward during those 6 minutes.
teacher: You've correctly identified that the product of -7 and 6 represents the total change in depth during the descent. How would you combine that change with the diver's initial depth of -18 meters to find their depth after the descent?
student: I think I add the change to the starting depth, so it would be \(-18 + (-42)\). I’m a little unsure with the two negative signs though.
teacher: You're starting at -18 meters and then adding a change of -42 meters. What does adding a negative number mean for the diver's depth?
student: It means the diver goes even deeper, so the depth becomes more negative. I think I need to combine \(-18\) and \(-42\), but I’m not sure of the final number yet.
teacher: You're right, the depth becomes more negative. What is the result when you combine -18 and -42?
student: \(-18 + (-42) = -60\), so the diver is at \(-60\) meters after the descent.
```

### 8a72eaca-f99f-4226-92d4-f73cc0492a2e:1:action:8

- Attempt/action: `8a72eaca-f99f-4226-92d4-f73cc0492a2e:1` / `8a72eaca-f99f-4226-92d4-f73cc0492a2e:1:action:8`; turn `8`; skill `Equation Solving Two or Fewer Steps`
- Transition: `generic -> generic`
- Baseline: `[generic=0.679809, probing=0.068721, focus=0.202380, telling=0.049090]`
- Candidate: `[generic=0.837584, probing=0.017068, focus=0.061508, telling=0.083840]`
- Delta telling/focus: `0.034750` / `-0.140872`
- Problem: A rectangular garden has a length that is 6 meters more than its width. The area of the garden is 72 m².

Find the width and length of the garden.

```text
teacher: How would you start solving this problem?
student: I would let the width be \(x\) meters. Then the length would be \(x+6\) meters. Since the area is 72 m², I would use the formula length × width = area to make an equation.
teacher: That's a great plan! What equation would you get when you put your expressions for length and width into the area formula?
student: i dont know
teacher: You've correctly identified that length × width = area. Can you try writing out what that looks like when you substitute your expressions for length and width into the formula?
student: The equation would be:

$$
x(x+6)=72
$$

because the width is \(x\) and the length is \(x+6\).
teacher: That's the correct equation! Now, what would be your next step to solve for \(x\)? How can you simplify and rearrange this equation?
student: I would expand the left side first:

$$
x(x+6)=x^2+6x
$$

So the equation becomes:

$$
x^2+6x=72
$$

Then move 72 to the left side:

$$
x^2+6x-72=0
$$

Now I can factor the quadratic to solve for \(x\).
teacher: You're ready to factor that quadratic. What's your approach for finding the factors of \(x^2+6x-72=0\)?
student: what?
teacher: What comes to mind when you think about factoring a quadratic equation like this one?
student: no idea
teacher: There are a few ways to factor quadratic equations. Do you recall any of them?
student: no
```

### bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:8

- Attempt/action: `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1` / `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:8`; turn `8`; skill `Multiplication and Division Integers`
- Transition: `focus -> focus`
- Baseline: `[generic=0.167329, probing=0.060054, focus=0.679405, telling=0.093212]`
- Candidate: `[generic=0.191028, probing=0.058509, focus=0.544370, telling=0.206092]`
- Delta telling/focus: `0.112880` / `-0.135035`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?

```text
teacher: How would you start solving this problem?
student: I’d probably start with the descent part first. Since it says \(-7\) meters per minute for 6 minutes, I think I need to multiply those somehow, but I’m not fully sure what to do with the starting \(-18\).
teacher: You're on the right track with multiplying -7 and 6. What would that product represent in terms of the diver's movement?
student: I think it represents how far the diver moved downward during those 6 minutes.
teacher: You've correctly identified that the product of -7 and 6 represents the total change in depth during the descent. How would you combine that change with the diver's initial depth of -18 meters to find their depth after the descent?
student: I think I add the change to the starting depth, so it would be \(-18 + (-42)\). I’m a little unsure with the two negative signs though.
teacher: You're starting at -18 meters and then adding a change of -42 meters. What does adding a negative number mean for the diver's depth?
student: It means the diver goes even deeper, so the depth becomes more negative. I think I need to combine \(-18\) and \(-42\), but I’m not sure of the final number yet.
teacher: You're right, the depth becomes more negative. What is the result when you combine -18 and -42?
student: \(-18 + (-42) = -60\), so the diver is at \(-60\) meters after the descent.
teacher: Great job finding the depth after the descent! Now, let's move on to the next part of the problem. The diver rises 36 meters at a rate of +9 meters per minute. What is the next question you need to answer based on this information?
student: I need to figure out how many minutes it takes to rise 36 meters at 9 meters per minute.
teacher: You're right, you need to find the time. What operation would you use with 36 meters and +9 meters per minute to calculate how many minutes the rise takes?
student: I think I should divide \(36\) by \(9\) to find the number of minutes.
```

### 5413cab2-22c8-4536-a66b-66619d887e84:1:action:11

- Attempt/action: `5413cab2-22c8-4536-a66b-66619d887e84:1` / `5413cab2-22c8-4536-a66b-66619d887e84:1:action:11`; turn `11`; skill `Equation Solving Two or Fewer Steps`
- Transition: `focus -> probing`
- Baseline: `[generic=0.349184, probing=0.139342, focus=0.401763, telling=0.109710]`
- Candidate: `[generic=0.122737, probing=0.336442, focus=0.268389, telling=0.272432]`
- Delta telling/focus: `0.162722` / `-0.133374`
- Problem: A rectangle has a length that is 4 cm more than its width. The area of the rectangle is 96 cm². Find the length and width of the rectangle.

```text
teacher: How would you start solving this problem?
student: I would start by letting the width be \(x\) cm. Then the length would be \(x + 4\) cm. Since the area is 96 cm², I can use the formula length × width = area to form an equation.
teacher: That's a great start! What equation would you form using those expressions for length and width?
student: I think the equation would be \(x + 4x = 96\), because the width is \(x\) and the length is \(x+4\).
teacher: You've correctly defined the width as \(x\) and the length as \(x+4\). Remember that the area of a rectangle is found by multiplying the length and the width. How would you write that multiplication using your expressions for length and width?
student: I would write it as \(x(x+4)=96\), because the width is \(x\) and the length is \(x+4\).
teacher: Excellent! Now that you have the equation \(x(x+4)=96\), what's the first step to simplify the left side of this equation?
student: i dont know
teacher: To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you do that?
student: i cannot understand clearly
teacher: No problem. Let's break it down. To simplify \(x(x+4)\), you need to multiply \(x\) by each term inside the parentheses. What do you get when you multiply \(x\) by the first term, which is \(x\)?
student: 2x
teacher: Not quite. When you multiply \(x\) by \(x\), you are multiplying a variable by itself. What is \(x \cdot x\) usually written as?
student: i really dont know
teacher: No problem. When you multiply a variable by itself, like \(x \cdot x\), we write it using an exponent. For example, \(2 \cdot 2\) is \(2^2\). How would you write \(x \cdot x\) using an exponent?
student: I think it would be \(x^2\).
teacher: Exactly! So, you've multiplied \(x\) by the first term, \(x\), to get \(x^2\). Now, what do you get when you multiply \(x\) by the second term inside the parentheses, which is \(4\)?
student: I get \(4x\), because \(x \times 4 = 4x\).
teacher: Great! You've found that \(x \cdot x = x^2\) and \(x \cdot 4 = 4x\). Now, how do you combine these two results to show the simplified form of \(x(x+4)\)?
student: The simplified form is \(x^2 + 4x\).
```

### bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:4

- Attempt/action: `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1` / `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:4`; turn `4`; skill `Multiplication and Division Integers`
- Transition: `focus -> focus`
- Baseline: `[generic=0.124376, probing=0.210824, focus=0.557339, telling=0.107462]`
- Candidate: `[generic=0.181356, probing=0.169438, focus=0.430609, telling=0.218598]`
- Delta telling/focus: `0.111136` / `-0.126730`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?

```text
teacher: How would you start solving this problem?
student: I’d probably start with the descent part first. Since it says \(-7\) meters per minute for 6 minutes, I think I need to multiply those somehow, but I’m not fully sure what to do with the starting \(-18\).
teacher: You're on the right track with multiplying -7 and 6. What would that product represent in terms of the diver's movement?
student: I think it represents how far the diver moved downward during those 6 minutes.
teacher: You've correctly identified that the product of -7 and 6 represents the total change in depth during the descent. How would you combine that change with the diver's initial depth of -18 meters to find their depth after the descent?
student: I think I add the change to the starting depth, so it would be \(-18 + (-42)\). I’m a little unsure with the two negative signs though.
```

### bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:5

- Attempt/action: `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1` / `bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:5`; turn `5`; skill `Multiplication and Division Integers`
- Transition: `focus -> telling`
- Baseline: `[generic=0.168604, probing=0.130409, focus=0.383115, telling=0.317872]`
- Candidate: `[generic=0.192592, probing=0.098485, focus=0.258171, telling=0.450751]`
- Delta telling/focus: `0.132880` / `-0.124944`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?

```text
teacher: How would you start solving this problem?
student: I’d probably start with the descent part first. Since it says \(-7\) meters per minute for 6 minutes, I think I need to multiply those somehow, but I’m not fully sure what to do with the starting \(-18\).
teacher: You're on the right track with multiplying -7 and 6. What would that product represent in terms of the diver's movement?
student: I think it represents how far the diver moved downward during those 6 minutes.
teacher: You've correctly identified that the product of -7 and 6 represents the total change in depth during the descent. How would you combine that change with the diver's initial depth of -18 meters to find their depth after the descent?
student: I think I add the change to the starting depth, so it would be \(-18 + (-42)\). I’m a little unsure with the two negative signs though.
teacher: You're starting at -18 meters and then adding a change of -42 meters. What does adding a negative number mean for the diver's depth?
student: It means the diver goes even deeper, so the depth becomes more negative. I think I need to combine \(-18\) and \(-42\), but I’m not sure of the final number yet.
```

### 8a72eaca-f99f-4226-92d4-f73cc0492a2e:1:action:6

- Attempt/action: `8a72eaca-f99f-4226-92d4-f73cc0492a2e:1` / `8a72eaca-f99f-4226-92d4-f73cc0492a2e:1:action:6`; turn `6`; skill `Equation Solving Two or Fewer Steps`
- Transition: `generic -> generic`
- Baseline: `[generic=0.755817, probing=0.053648, focus=0.156182, telling=0.034353]`
- Candidate: `[generic=0.898971, probing=0.012346, focus=0.040506, telling=0.048177]`
- Delta telling/focus: `0.013824` / `-0.115676`
- Problem: A rectangular garden has a length that is 6 meters more than its width. The area of the garden is 72 m².

Find the width and length of the garden.

```text
teacher: How would you start solving this problem?
student: I would let the width be \(x\) meters. Then the length would be \(x+6\) meters. Since the area is 72 m², I would use the formula length × width = area to make an equation.
teacher: That's a great plan! What equation would you get when you put your expressions for length and width into the area formula?
student: i dont know
teacher: You've correctly identified that length × width = area. Can you try writing out what that looks like when you substitute your expressions for length and width into the formula?
student: The equation would be:

$$
x(x+6)=72
$$

because the width is \(x\) and the length is \(x+6\).
teacher: That's the correct equation! Now, what would be your next step to solve for \(x\)? How can you simplify and rearrange this equation?
student: I would expand the left side first:

$$
x(x+6)=x^2+6x
$$

So the equation becomes:

$$
x^2+6x=72
$$

Then move 72 to the left side:

$$
x^2+6x-72=0
$$

Now I can factor the quadratic to solve for \(x\).
teacher: You're ready to factor that quadratic. What's your approach for finding the factors of \(x^2+6x-72=0\)?
student: what?
```

