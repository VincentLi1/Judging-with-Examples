# llm_grader


## Dataset Field Description
<img width="585" alt="image" src="https://github.com/user-attachments/assets/88838db8-f961-4d3d-91fb-ed15b3d76e1c" />

+ answer: is the student response to the current question.
+ sample_answer: refers to the correct reference answer, which is produced by a professional team of teachers and TAs and has been repeatedly verified as the correct answer in many years of classroom teaching.
+ sample_criteria: refers to the scoring guidelines that TAs use to score student responses, for example, what kind of answers will get what kind of scores.
+ full_points: refers to the full score of the current score.
+ score_1/2/3: refers to the scores from three different TAs.
+ score_outlier: is a field in the outlier.json file that some questions have, which refers to unreasonable scores. We regard the scores of professional TAs as reasonable scores, and add a score that is significantly different from the TAs' scores for some questions, which is defined as the score_outlier field. For details of this part, please refer to Section 3.3: Post-grading Review of the paper.

## About the types of questions collected
The questions in the tutorials we collected are mainly OS problem description (question field) + code to be executed (in the tutorialCode folder). For this type of question, students need to answer based on the question description and the results observed after running the code.

## About the specific method of human grader grading
+ Double-blind grading: TA does not know the student information, and TAs score independently.
+ Grading process: TAs participate in the entire teaching process of OS, from lecture to turtorial, and TAs are very familiar with the professional knowledge of this course. Before grading, TA will receive reference answers and document guidance on grading standards and a short training. TA also run code when grading.
+ Grading order: In order to ensure the accuracy and authority of human TA's grading, in actual grading, each question is scored by multiple TAs. These questions q1-q6 are selected from multiple tutorials in the entire OS course. For questions in the same tutorial, TA scores in the order of questions. Otherwise, it moves to student B. And our entire TAs team checked TA's scoring to ensure the rationality and correctness of the ground truth score.


## Citation
Please use the following when referencing [OS dataset](https://github.com/wenjing1170/llm_grader)
```
@article{xie2024grade,
  title={Grade like a human: Rethinking automated assessment with large language models},
  author={Xie, Wenjing and Niu, Juxin and Xue, Chun Jason and Guan, Nan},
  journal={arXiv preprint arXiv:2405.19694},
  year={2024}
}
```
