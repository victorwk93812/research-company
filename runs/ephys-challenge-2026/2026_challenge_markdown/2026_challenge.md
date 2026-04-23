# Spooky Action at a Distance in Quantum Circuits

### 1 Problem Summary

A "Bell state" is a quantum state of two qubits that features the maximum quantum entanglement between the two bits, which is what Einstein called "Spooky action at a distance". Design a quantum circuit that puts two distantly separated qubits into a Bell state.

## 2 The Problem

Consider a special Quantum Processing Unit (QPU), where the layout of its qubits forms a two-legged ladder shape (see Fig. 1). Two-qubit quantum gates (e.g., CNOT gates) on the processor can only act on two adjacent qubits. The solid lines in Fig. 1 connect adjacent qubits that can be acted upon by two-qubit quantum gates; that is to say, a two-qubit gate can act on the two qubits numbered e0 and 1, or 1 and 7 in the figure, but cannot act on the two qubits numbered 1 and 3, or e0 and 6.

![](assets/_page_0_Figure_7.jpeg)

Fig. 1

![](assets/_page_0_Picture_9.jpeg)

Fig. 2

• Design a quantum circuit to put the qubits e0 and e1 at the two ends of a single leg into any of the following Bell states:

$$\begin{split} |\Phi^{+}\rangle &= \frac{1}{\sqrt{2}} \big( |00\rangle + |11\rangle \big) \\ |\Phi^{-}\rangle &= \frac{1}{\sqrt{2}} \big( |00\rangle - |11\rangle \big) \\ |\Psi^{+}\rangle &= \frac{1}{\sqrt{2}} \big( |01\rangle + |10\rangle \big) \\ |\Psi^{-}\rangle &= \frac{1}{\sqrt{2}} \big( |01\rangle - |10\rangle \big) \end{split}$$

- The usable quantum gates are limited to single-qubit gates and adjacent two-qubit gates.
- This circuit should have horizontal (along the direction of the ladder legs) scalability, applicable to any length (L) of the two-legged ladder type qubit layout (see Fig. 2).

# 3 Competition Rules

- Eligibility for participation in this competition: Enrolled college, master's, or doctoral students nationwide.
- Participants can form teams of 1 to a maximum of 4 people; each person is limited to participating in 1 team.
- The competition process is divided into two stages: the Preliminary round and the Finals.
- Preliminary round submissions must be handed in online, and the submission must include:
  - \* A diagram and explanation of the quantum circuit.
  - \* An explanation of which Bell state the two target qubits are in: $|\Phi^+\rangle$ , $|\Phi^-\rangle$ , $|\Psi^+\rangle$ , or $|\Psi^-\rangle$ .
  - \* Solution verification methods, and/or derivations.
  - \* The code to execute the quantum circuit, annotated with required packages and execution instructions.
  - \* A brief description of the research, problem-solving, and learning process; if there are reference materials, advisor contributions, use of AI tools, or use of others' work, etc., they should be explicitly listed, and the scope of use explained.

#### • Finals Judging

Those who pass the preliminary round will be invited to participate in the finals presentation on 2026/5/16 (Saturday). On that day, the judging committee will select the winning teams and works based on merit.

### Competition Rewards

This event will select the following awards based on the correctness, innovation, rigor of the verification method/derivation, code quality, and clarity of the written explanation and oral presentation of the submitted works:

\* Grand Prize: NT$20,000 reward.

\* Innovation Award: NT$10,000 reward.

\* Honorable Mention: NT$5,000 reward.

#### Additionally, we will provide:

\* **Quick Draw Award**: A reward of **NT$500** will be awarded to the first 6 teams that complete registration the fastest and upload their works within the deadline. The evaluation time is based on the registration completion time, not the work upload time, but the latter must fall within the deadline.

### • Event Time and Location

- \* Before 2026/05/06, 23:59, complete registration and work upload. Registration and submission URL: https://forms.gle/bDQBQZ8aSJuHbzZE7 (Participants can register first within the deadline, then upload the work, and the work can be updated and re-uploaded).
- \* 2026/05/11: Announcement of shortlisted teams, notification to participate in the finals.
- \* **2026**/**05**/**16**: Finals, oral presentation and explanation of works by shortlisted teams. Location: Institute of Applied Physics, National Chengchi University (IIR Campus)
- The organizer will announce the shortlisted teams and winning teams on the webpage of the "Institute of Applied Physics, National Chengchi University".

# 4 Other Notices

- Competition awards will be selected by the judges at the finals presentation on 5/16 based on merit; if necessary, awards may be left vacant, adjusted, or added.
- The prize money includes tax and withholding will be processed according to the provisions of the Income Tax Act.
- The organizer reserves the right to final modification, alteration, interpretation, and cancellation of this event.
- The finals event will be video recorded and photographed, and its images will be used by the organizer and co-organizers for future educational promotion and outcome recording purposes.

### <span id="page-3-0"></span>Organizer:

Institute of Applied Physics, National Chengchi University
Undergraduate Program in Electron Physics, National Chengchi University

### Sponsors and Co-organizers:

NSTC "Quantum Virtual Machine" Project
NSTC "Physics Department Research Features Development" Project

![](assets/_page_3_Picture_6.jpeg)

Registration URL
