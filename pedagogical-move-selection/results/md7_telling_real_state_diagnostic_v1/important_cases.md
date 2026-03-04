# Important Case Inspection

Manual diagnostic labels are boundary judgments, not causal ground truth.

## Every real state changed to telling

### a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:6

- Group/skill: `Multiplication Whole Numbers`
- Baseline → candidate: `probing` → `telling`
- Baseline probabilities: `G=0.001/P=0.947/F=0.023/T=0.030`
- Candidate probabilities: `G=0.001/P=0.217/F=0.075/T=0.707`
- Δ telling / Δ focus: `0.67741` / `0.05245`
- Problem: A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks.

How many notebooks are there altogether?
- Prior dialogue:

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

- Group/skill: `Multiplication Whole Numbers`
- Baseline → candidate: `probing` → `telling`
- Baseline probabilities: `G=0.002/P=0.900/F=0.038/T=0.061`
- Candidate probabilities: `G=0.001/P=0.213/F=0.067/T=0.719`
- Δ telling / Δ focus: `0.65811` / `0.02974`
- Problem: A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks.

How many notebooks are there altogether?
- Prior dialogue:

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

### bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:6

- Group/skill: `Multiplication and Division Integers`
- Baseline → candidate: `focus` → `telling`
- Baseline probabilities: `G=0.201/P=0.120/F=0.371/T=0.308`
- Candidate probabilities: `G=0.079/P=0.113/F=0.324/T=0.483`
- Δ telling / Δ focus: `0.17524` / `-0.04718`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?
- Prior dialogue:

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

- Group/skill: `Multiplication and Division Integers`
- Baseline → candidate: `focus` → `telling`
- Baseline probabilities: `G=0.169/P=0.130/F=0.383/T=0.318`
- Candidate probabilities: `G=0.070/P=0.094/F=0.356/T=0.480`
- Δ telling / Δ focus: `0.16194` / `-0.02738`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?
- Prior dialogue:

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
## Every hard-negative candidate telling case

### I04

- Group/skill: `I_recoverable_after_one_scaffold`
- Baseline → candidate: `generic` → `telling`
- Baseline probabilities: `G=0.997/P=0.002/F=0.000/T=0.001`
- Candidate probabilities: `G=0.020/P=0.006/F=0.011/T=0.963`
- Δ telling / Δ focus: `0.96248` / `0.01033`
- Problem: Calculate (-5) × (-3).
- Prior dialogue:

```text
teacher: What answer do you get?
student: -15
teacher: The signs are the same; revisit the sign rule.
student: Same signs make positive, so 15?
```

### I03

- Group/skill: `I_recoverable_after_one_scaffold`
- Baseline → candidate: `generic` → `telling`
- Baseline probabilities: `G=0.996/P=0.002/F=0.000/T=0.001`
- Candidate probabilities: `G=0.001/P=0.002/F=0.047/T=0.950`
- Δ telling / Δ focus: `0.94933` / `0.04649`
- Problem: Simplify 5x + 2x.
- Prior dialogue:

```text
teacher: What is the result?
student: 7x²
teacher: You are adding like terms, not multiplying x by x.
student: Then it might be 7x.
```

### I01

- Group/skill: `I_recoverable_after_one_scaffold`
- Baseline → candidate: `generic` → `telling`
- Baseline probabilities: `G=0.547/P=0.447/F=0.003/T=0.003`
- Candidate probabilities: `G=0.001/P=0.004/F=0.436/T=0.559`
- Δ telling / Δ focus: `0.55529` / `0.43306`
- Problem: Solve x + 8 = 20.
- Prior dialogue:

```text
teacher: What would you undo first?
student: Add 8.
teacher: To reverse adding 8, which inverse operation could you use?
student: Maybe subtract 8?
```
## Telling-plausible cases candidate did not tell

_None._
## Largest 10 real-state telling-probability increases

### a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:6

- Group/skill: `Multiplication Whole Numbers`
- Baseline → candidate: `probing` → `telling`
- Baseline probabilities: `G=0.001/P=0.947/F=0.023/T=0.030`
- Candidate probabilities: `G=0.001/P=0.217/F=0.075/T=0.707`
- Δ telling / Δ focus: `0.67741` / `0.05245`
- Problem: A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks.

How many notebooks are there altogether?
- Prior dialogue:

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

- Group/skill: `Multiplication Whole Numbers`
- Baseline → candidate: `probing` → `telling`
- Baseline probabilities: `G=0.002/P=0.900/F=0.038/T=0.061`
- Candidate probabilities: `G=0.001/P=0.213/F=0.067/T=0.719`
- Δ telling / Δ focus: `0.65811` / `0.02974`
- Problem: A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks.

How many notebooks are there altogether?
- Prior dialogue:

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

### bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:4

- Group/skill: `Multiplication and Division Integers`
- Baseline → candidate: `focus` → `focus`
- Baseline probabilities: `G=0.124/P=0.211/F=0.557/T=0.107`
- Candidate probabilities: `G=0.068/P=0.097/F=0.492/T=0.343`
- Δ telling / Δ focus: `0.23547` / `-0.06549`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?
- Prior dialogue:

```text
teacher: How would you start solving this problem?
student: I’d probably start with the descent part first. Since it says \(-7\) meters per minute for 6 minutes, I think I need to multiply those somehow, but I’m not fully sure what to do with the starting \(-18\).
teacher: You're on the right track with multiplying -7 and 6. What would that product represent in terms of the diver's movement?
student: I think it represents how far the diver moved downward during those 6 minutes.
teacher: You've correctly identified that the product of -7 and 6 represents the total change in depth during the descent. How would you combine that change with the diver's initial depth of -18 meters to find their depth after the descent?
student: I think I add the change to the starting depth, so it would be \(-18 + (-42)\). I’m a little unsure with the two negative signs though.
```

### a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:5

- Group/skill: `Multiplication Whole Numbers`
- Baseline → candidate: `probing` → `probing`
- Baseline probabilities: `G=0.000/P=0.984/F=0.006/T=0.009`
- Candidate probabilities: `G=0.000/P=0.754/F=0.018/T=0.227`
- Δ telling / Δ focus: `0.21789` / `0.01218`
- Problem: A warehouse packs 48 boxes of notebooks. Each box contains 125 notebooks.

How many notebooks are there altogether?
- Prior dialogue:

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

- Group/skill: `Multiplication and Division Integers`
- Baseline → candidate: `focus` → `telling`
- Baseline probabilities: `G=0.201/P=0.120/F=0.371/T=0.308`
- Candidate probabilities: `G=0.079/P=0.113/F=0.324/T=0.483`
- Δ telling / Δ focus: `0.17524` / `-0.04718`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?
- Prior dialogue:

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

- Group/skill: `Multiplication and Division Integers`
- Baseline → candidate: `focus` → `telling`
- Baseline probabilities: `G=0.169/P=0.130/F=0.383/T=0.318`
- Candidate probabilities: `G=0.070/P=0.094/F=0.356/T=0.480`
- Δ telling / Δ focus: `0.16194` / `-0.02738`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?
- Prior dialogue:

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

### ecaa67e8-e212-4866-877f-9e2dd4cced40:1:action:9

- Group/skill: `Complementary and Supplementary Angles`
- Baseline → candidate: `focus` → `focus`
- Baseline probabilities: `G=0.247/P=0.207/F=0.372/T=0.175`
- Candidate probabilities: `G=0.255/P=0.151/F=0.348/T=0.246`
- Δ telling / Δ focus: `0.07102` / `-0.02352`
- Problem: Two angles are supplementary. One angle is 18° more than twice the other angle.

Find the measure of both angles.
- Prior dialogue:

```text
teacher: How would you start solving this problem?
student: i have no idea
teacher: You mentioned you have no idea how to start. Does the term 'supplementary angles' sound familiar to you?
student: no
teacher: No problem at all. What do you remember about angles in general?
student: I know angles are measured in degrees, and I think a straight line is \(180^\circ\), but I’m not sure how that relates to supplementary angles.
teacher: You're right that a straight line is 180 degrees. How might two angles relate to that 180 degrees?
student: Maybe the two angles add up to \(180^\circ\), like they could form a straight line together.
teacher: That's a great observation! Based on what you just said, how would you define 'supplementary angles'?
student: i dont konw what supplementary angles means
teacher: You just said that two angles might add up to 180 degrees. How does that idea relate to the term 'supplementary angles'?
student: i dont know,just tell me
teacher: You've correctly identified that two angles adding up to 180 degrees is a key idea. What makes it difficult to connect that idea to the term 'supplementary angles'?
student: i dont undersand
teacher: You've made a great connection that the two angles add up to 180 degrees. How can you use that idea to start setting up the problem?
student: You've made a great connection that the two angles add up to 180 degrees. How can you use that idea to start setting up the problem?
```

### 5413cab2-22c8-4536-a66b-66619d887e84:1:action:15

- Group/skill: `Equation Solving Two or Fewer Steps`
- Baseline → candidate: `generic` → `generic`
- Baseline probabilities: `G=0.519/P=0.069/F=0.273/T=0.139`
- Candidate probabilities: `G=0.391/P=0.034/F=0.377/T=0.199`
- Δ telling / Δ focus: `0.05987` / `0.10397`
- Problem: A rectangle has a length that is 4 cm more than its width. The area of the rectangle is 96 cm². Find the length and width of the rectangle.
- Prior dialogue:

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

### ecaa67e8-e212-4866-877f-9e2dd4cced40:1:action:8

- Group/skill: `Complementary and Supplementary Angles`
- Baseline → candidate: `focus` → `focus`
- Baseline probabilities: `G=0.224/P=0.311/F=0.323/T=0.141`
- Candidate probabilities: `G=0.257/P=0.241/F=0.317/T=0.185`
- Δ telling / Δ focus: `0.04348` / `-0.00642`
- Problem: Two angles are supplementary. One angle is 18° more than twice the other angle.

Find the measure of both angles.
- Prior dialogue:

```text
teacher: How would you start solving this problem?
student: i have no idea
teacher: You mentioned you have no idea how to start. Does the term 'supplementary angles' sound familiar to you?
student: no
teacher: No problem at all. What do you remember about angles in general?
student: I know angles are measured in degrees, and I think a straight line is \(180^\circ\), but I’m not sure how that relates to supplementary angles.
teacher: You're right that a straight line is 180 degrees. How might two angles relate to that 180 degrees?
student: Maybe the two angles add up to \(180^\circ\), like they could form a straight line together.
teacher: That's a great observation! Based on what you just said, how would you define 'supplementary angles'?
student: i dont konw what supplementary angles means
teacher: You just said that two angles might add up to 180 degrees. How does that idea relate to the term 'supplementary angles'?
student: i dont know,just tell me
teacher: You've correctly identified that two angles adding up to 180 degrees is a key idea. What makes it difficult to connect that idea to the term 'supplementary angles'?
student: i dont undersand
```

### bf4054b0-35ba-46ac-8cef-3d83a59f61c0:1:action:3

- Group/skill: `Multiplication and Division Integers`
- Baseline → candidate: `focus` → `focus`
- Baseline probabilities: `G=0.023/P=0.382/F=0.561/T=0.035`
- Candidate probabilities: `G=0.002/P=0.151/F=0.774/T=0.073`
- Δ telling / Δ focus: `0.03778` / `0.21358`
- Problem: A diver is at −18 meters below sea level. They descend at −7 meters per minute for 6 minutes, then rise 36 meters at a rate of +9 meters per minute.

What is the diver’s depth after the descent?
How many minutes does the rise take?
What is the diver’s final depth after the rise?
- Prior dialogue:

```text
teacher: How would you start solving this problem?
student: I’d probably start with the descent part first. Since it says \(-7\) meters per minute for 6 minutes, I think I need to multiply those somehow, but I’m not fully sure what to do with the starting \(-18\).
teacher: You're on the right track with multiplying -7 and 6. What would that product represent in terms of the diver's movement?
student: I think it represents how far the diver moved downward during those 6 minutes.
```
## Largest 10 real-state focus-probability decreases

### 0adabdf3-b861-4b0f-b802-451cc208dd0b:1:action:3

- Group/skill: `Pythagorean Theorem`
- Baseline → candidate: `focus` → `probing`
- Baseline probabilities: `G=0.165/P=0.253/F=0.559/T=0.023`
- Candidate probabilities: `G=0.002/P=0.977/F=0.019/T=0.002`
- Δ telling / Δ focus: `-0.02181` / `-0.53930`
- Problem: A right triangle has a hypotenuse of 13 cm and one shorter side of 5 cm.

Find the length of the other shorter side.
- Prior dialogue:

```text
teacher: How would you start solving this problem?
student: I think I use the Pythagorean theorem, but I’m not sure how to set it up since 13 is already the hypotenuse.
teacher: You're right that the Pythagorean theorem is the way to go! How does the hypotenuse relate to the other two sides in the formula?
student: I think the hypotenuse is \(c\), so \(13^2\) should go on the \(c^2\) side. The other two sides go on the other side.
```

### 5413cab2-22c8-4536-a66b-66619d887e84:1:action:3

- Group/skill: `Equation Solving Two or Fewer Steps`
- Baseline → candidate: `focus` → `probing`
- Baseline probabilities: `G=0.043/P=0.323/F=0.625/T=0.009`
- Candidate probabilities: `G=0.004/P=0.873/F=0.119/T=0.004`
- Δ telling / Δ focus: `-0.00543` / `-0.50547`
- Problem: A rectangle has a length that is 4 cm more than its width. The area of the rectangle is 96 cm². Find the length and width of the rectangle.
- Prior dialogue:

```text
teacher: How would you start solving this problem?
student: I would start by letting the width be \(x\) cm. Then the length would be \(x + 4\) cm. Since the area is 96 cm², I can use the formula length × width = area to form an equation.
teacher: That's a great start! What equation would you form using those expressions for length and width?
student: I think the equation would be \(x + 4x = 96\), because the width is \(x\) and the length is \(x+4\).
```

### 8a72eaca-f99f-4226-92d4-f73cc0492a2e:1:action:4

- Group/skill: `Equation Solving Two or Fewer Steps`
- Baseline → candidate: `focus` → `probing`
- Baseline probabilities: `G=0.111/P=0.389/F=0.490/T=0.009`
- Candidate probabilities: `G=0.012/P=0.857/F=0.127/T=0.004`
- Δ telling / Δ focus: `-0.00549` / `-0.36362`
- Problem: A rectangular garden has a length that is 6 meters more than its width. The area of the garden is 72 m².

Find the width and length of the garden.
- Prior dialogue:

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
```

### dffa3aac-68c5-4b98-b5cc-3bddd6e5f956:1:action:3

- Group/skill: `Simplifying Expressions positive exponents`
- Baseline → candidate: `focus` → `probing`
- Baseline probabilities: `G=0.069/P=0.317/F=0.605/T=0.008`
- Candidate probabilities: `G=0.003/P=0.719/F=0.275/T=0.003`
- Δ telling / Δ focus: `-0.00494` / `-0.33073`
- Problem: Simplify:

$$ 3x^2 \cdot 4x^3 $$

Write your final answer using a single exponent for \(x\).
- Prior dialogue:

```text
teacher: How would you start simplifying this expression?
student: I would start by multiplying the coefficients, \(3 \times 4\), and then combine \(x^2\) and \(x^3\) using the exponent rule for multiplying powers with the same base.
teacher: That's a great plan! Can you remind me what that exponent rule is?
student: I think when multiplying powers with the same base, you multiply the exponents, so maybe \(x^2 \cdot x^3 = x^6\). I’m not completely sure though—I sometimes mix up when to add and when to multiply the exponents.
```

### f3133173-8f14-44d9-b485-97a42dc45030:1:action:4

- Group/skill: `Conversion of Fraction Decimals Percents`
- Baseline → candidate: `probing` → `probing`
- Baseline probabilities: `G=0.027/P=0.617/F=0.349/T=0.007`
- Candidate probabilities: `G=0.003/P=0.946/F=0.050/T=0.002`
- Δ telling / Δ focus: `-0.00473` / `-0.29938`
- Problem: A school survey shows that \(\frac{3}{5}\) of the students prefer studying with digital tools.

Convert \(\frac{3}{5}\) into:

A decimal
A percent

Then, if there are 200 students in total, how many students prefer digital tools?
- Prior dialogue:

```text
teacher: How would you start solving this problem?
student: I would start by converting \(\frac{3}{5}\) to a decimal by dividing 3 by 5. Then I can turn that decimal into a percent by multiplying by 100.
teacher: That's a great plan! Can you explain why dividing the numerator by the denominator converts a fraction to a decimal?
student: I think dividing the numerator by the denominator works because the fraction bar means division. So \(\frac{3}{5}\) means \(3 \div 5\). I’m not totally sure though—I sometimes get confused about whether it should be \(3 \div 5\) or \(5 \div 3\).
teacher: You're right that the fraction bar means division. To help clarify, what do the numerator (the top number) and the denominator (the bottom number) represent in a fraction like \(\frac{3}{5}\) when you think about sharing or dividing?
student: The numerator is how many parts you have, and the denominator is how many equal parts the whole is divided into.
```

### 8a72eaca-f99f-4226-92d4-f73cc0492a2e:1:action:3

- Group/skill: `Equation Solving Two or Fewer Steps`
- Baseline → candidate: `probing` → `probing`
- Baseline probabilities: `G=0.073/P=0.540/F=0.382/T=0.005`
- Candidate probabilities: `G=0.005/P=0.900/F=0.093/T=0.001`
- Δ telling / Δ focus: `-0.00354` / `-0.28904`
- Problem: A rectangular garden has a length that is 6 meters more than its width. The area of the garden is 72 m².

Find the width and length of the garden.
- Prior dialogue:

```text
teacher: How would you start solving this problem?
student: I would let the width be \(x\) meters. Then the length would be \(x+6\) meters. Since the area is 72 m², I would use the formula length × width = area to make an equation.
teacher: That's a great plan! What equation would you get when you put your expressions for length and width into the area formula?
student: i dont know
```

### b9a3b98d-ab91-4e24-b3f7-39a0623831b9:1:action:3

- Group/skill: `Polynomial Factors`
- Baseline → candidate: `probing` → `probing`
- Baseline probabilities: `G=0.023/P=0.603/F=0.371/T=0.004`
- Candidate probabilities: `G=0.001/P=0.899/F=0.098/T=0.002`
- Δ telling / Δ focus: `-0.00199` / `-0.27231`
- Problem: Factor completely:

$$ x^2 + 9x + 20 $$

Then use your factors to find the values of \(x\) that make the expression equal to zero.
- Prior dialogue:

```text
teacher: How would you start solving this problem?
student: I think I need to find two numbers that multiply to 20 and add to 9, but I’m not fully sure how to pick them.
teacher: That's a great start! To help you pick them, what are some pairs of numbers that multiply to 20?
student: Maybe \(1\) and \(20\), \(2\) and \(10\), and \(4\) and \(5\).
```

### 5413cab2-22c8-4536-a66b-66619d887e84:1:action:4

- Group/skill: `Equation Solving Two or Fewer Steps`
- Baseline → candidate: `focus` → `probing`
- Baseline probabilities: `G=0.149/P=0.213/F=0.611/T=0.027`
- Candidate probabilities: `G=0.049/P=0.574/F=0.351/T=0.026`
- Δ telling / Δ focus: `-0.00184` / `-0.25913`
- Problem: A rectangle has a length that is 4 cm more than its width. The area of the rectangle is 96 cm². Find the length and width of the rectangle.
- Prior dialogue:

```text
teacher: How would you start solving this problem?
student: I would start by letting the width be \(x\) cm. Then the length would be \(x + 4\) cm. Since the area is 96 cm², I can use the formula length × width = area to form an equation.
teacher: That's a great start! What equation would you form using those expressions for length and width?
student: I think the equation would be \(x + 4x = 96\), because the width is \(x\) and the length is \(x+4\).
teacher: You've correctly defined the width as \(x\) and the length as \(x+4\). Remember that the area of a rectangle is found by multiplying the length and the width. How would you write that multiplication using your expressions for length and width?
student: I would write it as \(x(x+4)=96\), because the width is \(x\) and the length is \(x+4\).
```

### dffa3aac-68c5-4b98-b5cc-3bddd6e5f956:1:action:4

- Group/skill: `Simplifying Expressions positive exponents`
- Baseline → candidate: `focus` → `probing`
- Baseline probabilities: `G=0.037/P=0.293/F=0.653/T=0.017`
- Candidate probabilities: `G=0.003/P=0.596/F=0.395/T=0.006`
- Δ telling / Δ focus: `-0.01120` / `-0.25717`
- Problem: Simplify:

$$ 3x^2 \cdot 4x^3 $$

Write your final answer using a single exponent for \(x\).
- Prior dialogue:

```text
teacher: How would you start simplifying this expression?
student: I would start by multiplying the coefficients, \(3 \times 4\), and then combine \(x^2\) and \(x^3\) using the exponent rule for multiplying powers with the same base.
teacher: That's a great plan! Can you remind me what that exponent rule is?
student: I think when multiplying powers with the same base, you multiply the exponents, so maybe \(x^2 \cdot x^3 = x^6\). I’m not completely sure though—I sometimes mix up when to add and when to multiply the exponents.
teacher: You're right that it's easy to mix them up! Let's think about what \(x^2\) and \(x^3\) actually represent. What does \(x^2\) mean in expanded form?
student: \(x^2\) means \(x \times x\).
```

### f3133173-8f14-44d9-b485-97a42dc45030:1:action:6

- Group/skill: `Conversion of Fraction Decimals Percents`
- Baseline → candidate: `focus` → `probing`
- Baseline probabilities: `G=0.048/P=0.458/F=0.477/T=0.017`
- Candidate probabilities: `G=0.013/P=0.753/F=0.224/T=0.009`
- Δ telling / Δ focus: `-0.00773` / `-0.25327`
- Problem: A school survey shows that \(\frac{3}{5}\) of the students prefer studying with digital tools.

Convert \(\frac{3}{5}\) into:

A decimal
A percent

Then, if there are 200 students in total, how many students prefer digital tools?
- Prior dialogue:

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
```
