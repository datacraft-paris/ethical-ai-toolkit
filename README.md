# Welcome to the Ethical AI Toolkit project

- **Project name**: Ethical AI Toolkit
- **Library name**: ethicalai
- **Authors**: Datacraft, Ekimetrics, Danone, Telecom Paris
- **Description**: Open source Ethical AI toolkit

This project aims at centralizing relevant materials & resources about fairness and ethics.
It centralizes efforts made by a group of Data Scientists to uncover the topic:

- [Trustworthy AI guidelines](#trustworthy)
- [Benchathon](#benchathon)
- [Fairness workflow, revisited](#workflow)
- [Workshop slide deck](data/ia_ethique.pdf) that sums up the initiative until Nov 14th 2021 [in French]
- [Future & areas of development](#dev)

<a name="trustworthy"></a>
## Trustworthy AI guidelines

Part of the work conducted by this group aimed at making Data Science-ready a set of good practices related to Trustworthy AI.
It goes beyond the fairness aspect of it, which is the main focus of this repository. Please, visit this [link](https://datacraft-paris.github.io/trustworthyai/) to know more.

<a name="benchathon"></a>
## Benchathon

One step of this initiative was to assess the state of the art about fairness in order not to reinvent the wheel.
6 libraries were assessed:
- [Dalex](https://github.com/ModelOriented/DALEX)
- [AIF360](https://github.com/Trusted-AI/AIF360)
- [Shapash](https://github.com/MAIF/shapash)
- [Aequitas](https://github.com/dssg/aequitas)
- [What if tool](https://pair-code.github.io/)
- [Fairlearn](https://fairlearn.org)

The outcome of the assessment can be found [here](https://docs.google.com/spreadsheets/d/1Z071Ih9S7XYEcXBoX4k7SNoy6Z5DqbUM6htcpn1J_WU/edit?usp=sharing).
Note that Dalex does not appear in that document because it was discovered after its production.

*Note: the name benchathon comes from the concatenation of benchmark & hackathon. This is the name that was given to the day we spent together uncovering those libraries.*

<a name="workflow"></a>
## Fairness workflow, revisited

2 main conclusions came out of the benchathon:
- Different libraries come with different advantages (Dalex is user-friendly and well-designed, when AIF360 implements more bias mitigation techniques e.g.). However, they do not easily interface together. **There is room for improvement**.
- Most of those libraries put a focus on the tools/methodologies they make available, slightly less on the "how to properly use them in the context of a real-life case?". **There is room for improvement**

This is the reason why we decided to come up with our own [experimental notebook](notebooks/ethical_ai.ipynb), that aims at:
- Reproducing a fairness workflow, highly based on already available material
- Emphasize what questions should be asked and by whom along the way
- Emphasize how to mix libraries

<a name="dev"></a>
## Areas of improvement

The current state of investigation mainly focuses on tabular classification. Other areas are still to be uncovered, including but not limited to:
- Regression
- Complex data structures like text or images
- Automatic bias detection

If you want to contribute, please feel free to raise an [issue](https://github.com/datacraft-paris/ethical-ai-toolkit/issues) or contact [Xavier](xavier.lioneton@datacraft.paris>)