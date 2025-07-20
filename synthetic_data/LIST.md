
### MCQA
Rule: remove if len(input) > 768 or len(output) > 512
 - [x] MMLU + instruction + bench-format + synthetic-explanation
 - [x] SciQ + instruction + bench-format + synthetic-explanation
 - [x] ARC + instruction + bench-format + synthetic-explanation
 - [x] nvidia/OpenScience + instruction + bench-format

### CHAT
Rule: remove if len(input) > 768 or len(output) > 512
 - [ ] HuggingFaceTB/smoltalk
 - [ ] teknium/OpenHermes-2.5
 - [ ] microsoft/orca-agentinstruct-1M-v1
 - [ ] IgnoraZ/SynthQuestions -> synthquestions
 - [ ] IgnoraZ/SynthQuestions -> realquestions
 - [ ] TIGER-Lab/WebInstructSub

### Math/Reasoning
Rule: remove if len(input) > 768 or len(output) > 512
 - [ ] facebook/natural_reasoning
 - [ ] math-ai/StackMathQA -> stackmathqafull-1q1a
 - [ ] meta-math/MetaMathQA + instruction + bench-format
 - [ ] TIGER-Lab/MATH-plus + instruction + bench-format
 - [ ] nvidia/OpenMathInstruct-2 + instruction + bench-format

### Web Text
 - [ ] math-ai/AutoMathText