from parlai.core.teachers import DialogTeacher
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt

from typing import Optional

from .build import build
from loguru import logger
from pathlib import Path
import random
import json


def split_qa(qa_text):

    if "?" not in qa_text:
        logger.error(f"`?` not found in question and answer pair")
        return None

    qm_index = qa_text.index("?")

    question = qa_text[: qm_index + 1]
    answer = qa_text[qm_index + 1 :]

    return (question, answer)


class InferenceGuidedDialogueTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.id = "inf_dial"
        self.datatype = opt['datatype']
        dpath = build(opt)
        if opt['datatype'].startswith("test"):
            opt['datafile'] = Path(dpath) / "all_test_responses.json"
        else:
            opt['datafile'] = Path(dpath) / "all_train_responses.json"
        self.generation_target = opt.get("generation_target")
        self.no_special_tokens = opt.get("no_special_tokens")
        self.generate_full_sequence = opt.get("generate_full_sequence")

        super().__init__(opt, shared)

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Inference-Guided Dialogue Arguments')
        agent.add_argument(
            '-gt',
            '--generation_target',
            type=str,
            default="response",
            choices=['response', 'infq_aresponse', 'infqa_response'],
            help='Targets to use for generation. Refer to README for more information: parlai/tasks/inference_guided_dialogue/README.md',
        )

        agent.add_argument(
            '-nosp',
            '--no_special_tokens',
            action="store_true",
            help="Don't use any special tokens such as <speaker1> etc.",
        )

        agent.add_argument(
            '-gf',
            '--generate_full_sequence',
            action="store_true",
            help="Generate the full sequence of question-answer-response instead of generating at each separate turn.",
        )

        return parser

    def setup_data(self, path):

        """
        Example:
        <speaker1> Cameron knew I was dishonest! \n
        <speaker2> You should have behaved better! \n
        <speaker1> Hmm. Cameron now know when I lied to him. \n
        <infq> How to describe <speaker1>? <infa> Unable to be trusted, deceitful \n
        <speaker2> I cant believe you lied to your friend.
        """

        print(f"Loading: {path}")
        with open(path, "r") as f:
            data_ = json.load(f)

        pct90_index = int(len(data_) * 9 / 10)
        # use 9:1 split
        if self.opt['datatype'].startswith("train"):
            data_ = data_[:pct90_index]
            random.shuffle(data_)

        elif self.opt['datatype'].startswith("valid"):
            data_ = data_[pct90_index:]

        processed_data = []
        for d in data_:
            dial_hist = d['utterance']
            question, answer = split_qa(d["triple_NL"])
            question, answer = question.strip(), answer.strip()
            response = d['response'].strip()
            if not self.no_special_tokens:
                speaker_label = "<speaker2>" if len(dial_hist) % 2 else "<speaker1>"
                response = f"{speaker_label} {response}"

            processed = []
            for idx, turn in enumerate(dial_hist):
                turn = turn.strip()
                if self.no_special_tokens:
                    processed.append(turn)
                else:
                    speaker_label = "<speaker2>" if idx % 2 else "<speaker1>"
                    processed.append(f"{speaker_label} {turn}")

            if self.no_special_tokens:
                processed_inf_q = question
                processed_inf_a = answer
            else:
                processed_inf_q = f"<infq> {question}"
                processed_inf_a = f"<infa> {answer}"

            # default: generate only the response
            if self.generation_target == "response":
                input_text = '\n'.join(processed)
                output_text = [response]  # provide as list of candidates
                new_episode = True
                processed_data.append((input_text, output_text, new_episode))

            else:
                # generate the full sequence in one go
                if self.generate_full_sequence:
                    if self.generation_target == "infqa_response":
                        input_text = '\n'.join(processed)
                        output_text = [
                            f"{processed_inf_q} {processed_inf_a} {response}"
                        ]
                        new_episode = True

                    elif self.generation_target in ["infq_aresponse"]:
                        input_text = '\n'.join(processed + [processed_inf_q])
                        output_text = [
                            f"{processed_inf_a} {response}"
                        ]  # provide as list of candidates
                        new_episode = True

                    processed_data.append((input_text, output_text, new_episode))

                # generate through multiple turns
                else:
                    # add question generation turn
                    if self.generation_target == "infqa_response":
                        input_text = '\n'.join(processed)
                        output_text = [processed_inf_q]
                        new_episode = True
                        processed_data.append((input_text, output_text, new_episode))

                    # add question answer generation turn
                    if self.generation_target in ["infq_aresponse", "infqa_response"]:
                        input_text = '\n'.join(processed + [processed_inf_q])
                        output_text = [processed_inf_a]  # provide as list of candidates
                        new_episode = True
                        processed_data.append((input_text, output_text, new_episode))

                    # add response generation turn
                    input_text = '\n'.join(
                        processed + [processed_inf_q, processed_inf_a]
                    )
                    output_text = [response]  # provide as list of candidates
                    new_episode = True
                    processed_data.append((input_text, output_text, new_episode))

        if self.datatype == "train":
            random.shuffle(processed_data)

        for it, ot, new_ep in processed_data:
            yield {
                "text": it,
                "labels": ot,
            }, new_ep


class DefaultTeacher(InferenceGuidedDialogueTeacher):
    pass
