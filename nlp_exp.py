import dataclasses
import json
from threading import Thread
from time import sleep
import time
import os
import traceback

from tqdm import tqdm
import torch
import plwn
import numpy as np
from dataclasses import dataclass
from multiprocessing import Process, Manager
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from enum import Enum
from transformers import AutoTokenizer, AutoModel, LlamaTokenizerFast
from langchain.embeddings import LlamaCppEmbeddings

QUEUE_SIZE = 60
LLAMA_LOCAL_PATH = "llama/llama-2-7b-chat"
DATASET_LOCAL_PATH = "datasets"
EMBBEDINGS_PATH = "datasets/embeddings"
RESULTS_PATH = "datasets/results"
PLWN_API_PATH = "plwn-api_plwn_dump_new_07-12-2022.sqlite"


class Dataset:
    def __init__(self, lang, name, path):
        self.lang = lang
        self.name = name
        self.path = path
        self.lines = []
        self.dataset_size = 0


@dataclass
class WordsRelated:
    position: tuple
    related: dict
    unrelated: dict


@dataclass
class WordSynonyms:
    position: tuple
    other_synset_syn: list
    same_synset_syn: list


@dataclass
class InputSentence:
    sentence: str
    lemmas: list
    orth: list
    positions: list
    synset_ids: list
    index: int
    lang: str = "pl"


@dataclass
class CountedSentence:
    idx: int
    sims_same: list
    sims_other: list
    nbr_words: int
    word_sims_same: list
    word_sims_other: list


class EmbeddingModelType(Enum):
    SENT_BERT = "sent-bert"
    BERT = "bert"
    LABSE = "labse"
    LLAMA = "llama"
    LLAMA_HF = "llama-hf"


class EmbeddingModel:

    def __init__(self, model, tokenizer, model_type=EmbeddingModelType.SENT_BERT):
        if model_type != EmbeddingModelType.LLAMA:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = tokenizer
        if model_type not in [EmbeddingModelType.LLAMA,
                              EmbeddingModelType.LLAMA_HF]:
            self.model = model.to(self.device)
            if tokenizer:
                self.tokenizer = tokenizer
        else:
            self.model = model
        self.model_type = model_type

    def get_embeddings(self, sentence):
        if isinstance(sentence, str):
            sentence = [sentence]

        if self.model_type in [EmbeddingModelType.LLAMA]:
            result = torch.Tensor()
            for sent in sentence:
                result = torch.cat((result, torch.Tensor(self.model.embed_query(sent)).unsqueeze(0)), 0)
            return result
        elif self.model_type == EmbeddingModelType.LLAMA_HF:
            result = torch.Tensor()
            with torch.no_grad():
                for sent in sentence:
                    seq_ids = self.tokenizer(sent, return_tensors='pt')["input_ids"]
                    result = torch.cat((result,
                                    self.model(seq_ids)["last_hidden_state"].mean(axis=[0,1]).unsqueeze(0).cpu()),
                                    0)
            return result
        else:
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True)
            if self.model_type != EmbeddingModelType.LLAMA:
                inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = self.get_emb_by_model_type(outputs, inputs)
            return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_emb_by_model_type(self, outputs, inputs):
        if self.model_type == EmbeddingModelType.SENT_BERT:
            return self.mean_pooling(outputs, inputs["attention_mask"])
        else:
            return outputs.pooler_output

    def get_sent_words_embeddings(self, sentence, words_list):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is None")
        else:
            word_positions, sent_input_ids = self._get_words_positions(
                sentence, words_list
            )
            outputs = self._get_sent_embeddings(sent_input_ids)
            sent_embeddings = self.get_emb_by_model_type(outputs, sent_input_ids)

            word_state = outputs.last_hidden_state.cpu().squeeze(0)
            word_embeddings = self._get_words_embeddings(word_positions, word_state)
        return sent_embeddings, word_embeddings

    def get_words_embeddings(self, sentence, words_list):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is None")
        else:
            word_positions, sent_input_ids = self._get_words_positions(
                sentence, words_list
            )
            outputs = self._get_sent_embeddings(sent_input_ids)
            word_state = outputs.last_hidden_state.cpu().squeeze(0)
            word_embeddings = self._get_words_embeddings(word_positions, word_state)
        return word_embeddings

    def _get_words_positions(self, sentence, words_list):
        sent_input_ids = self.tokenizer(sentence, return_tensors="pt", padding=True)['input_ids']
        decoded = ["".join(self.tokenizer.decode(token).split(" ")) for token in sent_input_ids[0].tolist()]
        words_list = [word.lower() for word in words_list if word.isalpha()]
        token_to_pos = {i: token for i, token in enumerate(decoded) if token.isalpha()}
        word_positions = []
        while len(words_list) > 0:
            word = words_list.pop(0)
            match = [pos for pos, token in token_to_pos.items() if word.startswith(token.lower())]
            for pos, token in token_to_pos.items():
                if word.startswith(token.lower()):
                    word_positions.append((word, pos))
                    token_to_pos.pop(pos)
                    break
                elif len(match) > 0:
                    word_positions.append((word, match[0]))
                    break
                else:
                    word_positions.append((word, -1))
                    break
        return word_positions, sent_input_ids

    def _get_words_embeddings(self, word_positions, word_state):
        word_embeddings = torch.Tensor()
        for _, pos in word_positions:
            if pos != -1:
                word_embeddings = torch.cat(
                    (word_embeddings, word_state[pos].unsqueeze(0)), 0
                )
            else:
                if self.model_type == EmbeddingModelType.LLAMA_HF:
                    word_embeddings = torch.cat((word_embeddings, torch.zeros(1, 4096)), 0)
                else:
                    word_embeddings = torch.cat((word_embeddings, torch.zeros(1, 768)), 0)
        return word_embeddings

    def _get_sent_embeddings(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.device))
        return outputs

    def pairwise_similarity(
        self,
        org_embedding,
        changed_sentences,
    ):
        if len(changed_sentences) > 0:
            changed_embeddings = self.get_embeddings(changed_sentences)
            if len(org_embedding.shape) < 2:
                org_embedding = org_embedding.unsqueeze(0)
            if len(changed_embeddings.shape) < 2:
                changed_embeddings = changed_embeddings.unsqueeze(0)
            sims = pairwise_cosine_similarity(
                org_embedding.cpu(), changed_embeddings.cpu()).cpu().numpy()
            sims = sims.astype(np.float64)
            return 1 - sims

    def pairwise_words_similarity(
        self,
        org_sentence_with_words_list,
        changed_sentences_with_words_lists,
    ):
        org_sentence, org_words_list = org_sentence_with_words_list
        changed_sentences, changed_words_list = changed_sentences_with_words_lists

        org_embedding = self.get_words_embeddings(org_sentence, org_words_list)

        sims = []
        for changed_sentence, changed_words in zip(
            changed_sentences, changed_words_list
        ):
            similarity = self.words_similarity(
                (org_embedding, org_words_list),
                (changed_sentence, changed_words),
                embed_original=False,
            )
            sims.append(similarity)

        return np.array(sims)

    def sent_similarity(
        self,
        sentence_1,
        sentence_2,
    ):
        if len(sentence_1) > 0 and len(sentence_2) > 0:
            sent_1 = self.get_embeddings(sentence_1)
            sent_2 = self.get_embeddings(sentence_2)
            sims = pairwise_cosine_similarity(sent_1, sent_2).cpu().numpy()
            sims = sims.astype(np.float64)
            return sims

    def words_similarity(
        self,
        sentence_with_words_list_1,
        sentence_with_words_list_2,
        embed_original=True,
    ):
        sentence_1, words_list_1 = sentence_with_words_list_1
        sentence_2, words_list_2 = sentence_with_words_list_2

        if len(words_list_1) > 0 and len(words_list_2) > 0:
            if embed_original:
                word_emb_1 = self.get_words_embeddings(sentence_1, words_list_1)
            else:
                word_emb_1 = sentence_1
            word_emb_2 = self.get_words_embeddings(sentence_2, words_list_2)
            similarites = np.array([])
            for emb_1, emb_2 in zip(word_emb_1, word_emb_2):
                sims_word = pairwise_cosine_similarity(
                    torch.reshape(emb_1, (1, emb_1.shape[0])), 
                    torch.reshape(emb_2, (1, emb_2.shape[0]))
                    ).cpu().numpy()
                sims_word = sims_word.astype(np.float64)
                mean_sim = np.mean(sims_word)
                if np.isnan(mean_sim):
                    mean_sim = 0
                similarites = np.append(similarites, mean_sim)
            return similarites


def data_producer(queue_out, lines, queue_log, lang):
    """Producer function for data pipeline."""
    try:
        for idx, line in tqdm(enumerate(lines), total=len(lines)):
            if lang == "pl":
                text = line["text"]
                tokens = line["tokens"]
                lemmas = tokens["lemma"]
                orth = tokens["orth"]
                indices = line["wsd"]["index"]
                synset_ids = line["wsd"]["plWN_syn_id"]
                lemmas = [lemmas[i] for i in indices]
                positions = [(pos[0], pos[1]) for pos in tokens["position"]]
                positions = [positions[i] for i in indices]
                queue_out.put(InputSentence(text, lemmas, orth, positions, synset_ids, idx, lang))
            else:  # en
                queue_out.put(InputSentence(**line))
        queue_log.put("Finished producing data")
    except Exception as e:
        queue_log.put(f"Error in data producer: {e}")


def get_uuid(wn, lemma, variant):
    for syn in wn.synsets(lemma):
        variants = [lex.variant for lex in syn.lexical_units]
        if variant in variants:
            return syn.uuid
    return None


def data_saver(queue_in, queue_log, output_file, dataset_size, queues_to_kill):
    """Saver function for data pipeline."""
    try:
        item = 1
        count = 0
        if os.path.exists(output_file):
            os.remove(output_file)
        while item is not None:
            item = queue_in.get()
            if item is not None:
                if item.nbr_words > 0:
                    with open(output_file, "a") as f:
                        f.write(json.dumps(dataclasses.asdict(item)) + "\n")
                    count += 1
            if count == dataset_size - 1:
                queue_log.put("End of Saving...")
                break
        [q.put(None) for _ in range(10) for q in queues_to_kill]
    except Exception as e:
        queue_log.put(f"Error in data saver: {e}")
        queue_in.put(None)


def get_sent_model():
    sent_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    sent_model = AutoModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    return sent_model, sent_tokenizer


def get_bert_model(lang="pl"):
    if lang == "pl":
        bert_tokenizer = AutoTokenizer.from_pretrained(
            "allegro/herbert-klej-cased-tokenizer-v1"
        )
        bert_model = AutoModel.from_pretrained("allegro/herbert-klej-cased-v1")
    else:
        bert_tokenizer = AutoTokenizer.from_pretrained(
            "xlm-roberta-base"
        )
        bert_model = AutoModel.from_pretrained("xlm-roberta-base")
    return bert_model, bert_tokenizer


def get_labse_model():
    from transformers import BertTokenizerFast, BertModel

    labse_tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    labse_model = BertModel.from_pretrained("setu4993/LaBSE")
    return labse_model, labse_tokenizer


def get_llama_model():
    llama_model_path = f"{LLAMA_LOCAL_PATH}/ggml-model-f16.gguf"
    embeddings = LlamaCppEmbeddings(
        model_path=llama_model_path,
        n_ctx=1024,
        n_batch=1024,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False,
    )
    return embeddings, None


def get_llama_hf_model():

    llama_model_path = LLAMA_LOCAL_PATH
    full_model = AutoModel.from_pretrained(llama_model_path,
                                           torch_dtype=torch.float16,
                                           device_map="auto")
    full_model.eval()
    tokenizer = LlamaTokenizerFast.from_pretrained(llama_model_path)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return full_model, tokenizer


def load_model(name, lang):
    if name == EmbeddingModelType.SENT_BERT.value:
        return get_sent_model()
    elif name == EmbeddingModelType.BERT.value:
        return get_bert_model(lang)
    elif name == EmbeddingModelType.LABSE.value:
        return get_labse_model()
    elif name == EmbeddingModelType.LLAMA.value:
        return get_llama_model()
    elif name == EmbeddingModelType.LLAMA_HF.value:
        return get_llama_hf_model()


def count_similarity(queue_in, queue_out, queue_log, model, emb_pkl, lang):
    emb_array = torch.load(emb_pkl)
    model_loaded, tok = load_model(model, lang)
    model_type = EmbeddingModelType(model)
    model = EmbeddingModel(model_loaded, tok, model_type=model_type)
    queue_log.put(
        f"Device " f"{torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )
    try:
        item = 1
        while item is not None:
            item = queue_in.get()
            if item is not None:
                org_sent, idx, substitutions, orths, mode = item
                if substitutions:
                    try:
                        if mode in "synonyms":
                            sub_same_syn = ""
                            sub_other_syn = ""
                            begin = 0
                            for sub in substitutions:
                                sub_same_syn += (
                                    org_sent[begin : sub.position[0]]
                                    + sub.same_synset_syn[0]
                                )
                                sub_other_syn = (
                                    org_sent[begin : sub.position[0]]
                                    + sub.other_synset_syn[0]
                                )
                                begin = sub.position[1]
                            sub_same_syn += org_sent[begin:]
                            sub_other_syn += org_sent[begin:]
                            emb_org = emb_array[idx]
                            sims = model.pairwise_similarity(
                                emb_org, [sub_same_syn, sub_other_syn]
                            )
                            queue_out.put(
                                CountedSentence(
                                    idx, sims[0][0], sims[0][1], len(substitutions), [], []
                                )
                            )
                        elif mode == "words":
                            sub_same_syn = ""
                            sub_other_syn = ""
                            begin = 0
                            sub = substitutions[0]
                            same = [s for s in sub.same_synset_syn if s.isalpha()]
                            other = [s for s in sub.other_synset_syn if s.isalpha()]
                            if len(same) == 0 or len(other) == 0:
                                queue_out.put(CountedSentence(idx, [], [], 0, [], []))
                                continue
                            same = same[0]
                            other = other[0]
                            sub_same_syn += org_sent[begin: sub.position[0]] + same
                            sub_other_syn += org_sent[begin: sub.position[0]] + other
                            begin = sub.position[1]
                            sub_same_syn += org_sent[begin:]
                            sub_other_syn += org_sent[begin:]

                            orths_same_syn = orths.copy()
                            orths_same_syn[
                                orths.index(org_sent[sub.position[0]: sub.position[1]])
                            ] = same
                            orths_other_syn = orths.copy()
                            orths_other_syn[orths.index(org_sent[sub.position[0]:sub.position[1]])] = other
                            same_syn_idx_alpha = [
                                word for word in orths_same_syn if word.isalpha()
                            ].index(same)
                            other_syn_idx_alpha = [
                                word for word in orths_other_syn if word.isalpha()
                            ].index(other)
                            sims_same_syn = model.words_similarity(
                                (sub_same_syn, orths_same_syn), (org_sent, orths)
                            )
                            sims_other_syn = model.words_similarity(
                                (sub_other_syn, orths_other_syn), (org_sent, orths)
                            )
                            sims_same_per_distance = []
                            sims_other_per_distance = []

                            for dist in range(max(len(sims_same_syn[:same_syn_idx_alpha]), len(sims_same_syn[same_syn_idx_alpha:]))):
                                if dist == 0:
                                    sims_same_per_distance.append(
                                        sims_same_syn[same_syn_idx_alpha]
                                    )
                                else:
                                    left = same_syn_idx_alpha - dist
                                    right = same_syn_idx_alpha + dist
                                    mean_neighbours = []
                                    if left >= 0:
                                        mean_neighbours.append(sims_same_syn[left])
                                    if right < len(sims_same_syn):
                                        mean_neighbours.append(sims_same_syn[right])
                                    sims_same_per_distance.append(np.mean(mean_neighbours))


                            for dist in range(max(len(sims_other_syn[:other_syn_idx_alpha]), len(sims_other_syn[other_syn_idx_alpha:]))):
                                if dist == 0:
                                    sims_other_per_distance.append(
                                        sims_other_syn[other_syn_idx_alpha]
                                    )
                                else:
                                    left = other_syn_idx_alpha - dist
                                    right = other_syn_idx_alpha + dist
                                    mean_neighbours = []
                                    if left >= 0:
                                        mean_neighbours.append(sims_other_syn[left])
                                    if right < len(sims_other_syn):
                                        mean_neighbours.append(sims_other_syn[right])
                                    sims_other_per_distance.append(np.mean(mean_neighbours))

                            queue_out.put(
                                CountedSentence(
                                    idx,
                                    sims_same_per_distance,
                                    sims_other_per_distance,
                                    len(substitutions),
                                    [],
                                    [],
                                )
                            )

                        elif mode == "related":
                            sub = substitutions[0]
                            related = list(sub.related.items())[0]
                            unrelated = list(sub.unrelated.items())[0]

                            sub_related = (
                                org_sent[0: related[0][0]]
                                + related[1][0]
                                + org_sent[related[0][1]:]
                            )
                            sub_unrelated = (
                                org_sent[0: unrelated[0][0]]
                                + unrelated[1][0]
                                + org_sent[unrelated[0][1]:]
                            )
                            word = org_sent[sub.position[0]:sub.position[1]]
                            orths_related = orths.copy()
                            orths_related[
                                orths.index(org_sent[related[0][0]:related[0][1]])
                            ] = related[1][0]
                            orths_unrelated = orths.copy()
                            orths_unrelated[
                                orths.index(org_sent[unrelated[0][0]:unrelated[0][1]])
                            ] = unrelated[1][0]
                            rel_idx_alpha = [
                                word for word in orths_related if word.isalpha()
                            ].index(word)
                            unrel_idx_alpha = [
                                word for word in orths_unrelated if word.isalpha()
                            ].index(word)

                            sims_rel = model.words_similarity(
                                (sub_related, orths_related), (org_sent, orths)
                            )
                            sims_unrel = model.words_similarity(
                                (sub_unrelated, orths_unrelated), (org_sent, orths)
                            )
                            queue_out.put(
                                CountedSentence(
                                    idx,
                                    sims_rel[rel_idx_alpha],
                                    sims_unrel[unrel_idx_alpha],
                                    len(substitutions),
                                    [],
                                    [],
                                )
                            )
                        else:
                            raise ValueError("Wrong mode")
                    except Exception as e:
                        queue_log.put(f"Error in similarity inner: {e}")
                        traceback.print_exc()
                        queue_out.put(CountedSentence(idx, [], [], 0, [], []))
                else:
                    queue_out.put(CountedSentence(idx, [], [], 0, [], []))
    except Exception as e:
        queue_log.put(f"Error in similiarity: {e}")
        queue_in.put(None)


class Experiment:
    def __init__(self, plwn_model):
        self.wn = plwn.load(plwn_model)

    def word_synonyms(self, word, synset_id):
        synonyms = []
        for syn in self.wn.synsets(lemma=word):
            if syn.uuid != synset_id:
                for lex in syn.lexical_units:
                    if lex.lemma != word and " " not in lex.lemma:
                        synonyms.append(lex.lemma)
        return synonyms

    def word_from_same_synset(self, word, synset_id):
        synset_synonyms = []
        syn = self.wn.synset_by_id(synset_id)
        for lex in syn.lexical_units:
            if lex.lemma != word and " " not in lex.lemma:
                synset_synonyms.append(lex.lemma)
        return synset_synonyms

    def get_substitutions(self, words, synset_ids, indices, nbr_words=1):
        substitutions = []
        words_done = 0
        for word, synset_id, idx in zip(words, synset_ids, indices):
            try:
                same_synset = self.word_from_same_synset(word, synset_id)
                synset_synonyms = self.word_synonyms(word, synset_id)
            except Exception as e:
                print(e)
                continue
            if same_synset and synset_synonyms:
                words_done += 1
                substitutions.append(WordSynonyms(idx, synset_synonyms, same_synset))
            if words_done >= nbr_words:
                break
        return substitutions

    def get_relations(self, words, indices, synset_ids, synset_id, word):
        syn = self.wn.synset_by_id(synset_id)
        related = []
        unrelated = []
        for lex in syn.lexical_units:
            rel = lex.related()
            rel = [
                r.lemma
                for r in rel
                if r.lemma != lex.lemma and r.lemma in words and " " not in r.lemma
            ]
            rel_with_ids = [
                (w, i, s)
                for w, i, s in zip(words, indices, synset_ids)
                if w in rel and (w, i, s) not in related and w.isalpha()
            ]
            related.extend(rel_with_ids)
            unrelated.extend(
                [
                    (w, i, s)
                    for w, i, s in zip(words, indices, synset_ids)
                    if w not in rel and w != word and (w, i, s) not in unrelated and w.isalpha()
                ]
            )
        return related, unrelated

    def get_related_unrelated(self, words, synset_ids, indices, nbr_words=1):
        substitutions = []
        related, unrelated = {}, {}
        words_done = 0
        for word, synset_id, idx in zip(words, synset_ids, indices):
            try:
                if word.isalpha():
                    related, unrelated = self.get_relations(
                        words, indices, synset_ids, synset_id, word
                    )
                    related = {
                        tuple(pos): self.word_from_same_synset(word, synset_id)
                        for word, pos, synset_id in related[:nbr_words]
                    }
                    related = {k: v for k, v in related.items() if v and k != idx}
                    unrelated = {
                        tuple(pos): self.word_from_same_synset(word, synset_id)
                        for word, pos, synset_id in unrelated[:nbr_words]
                    }
                    unrelated = {k: v for k, v in unrelated.items() if v}
            except Exception as e:
                print(e)
                continue
            if related and unrelated:
                words_done += 1
                substitutions.append(WordsRelated(idx, related, unrelated))
            if words_done >= nbr_words:
                break
        return substitutions


def spoil_queue(queue_in, queue_out, queue_log, nbr_words, wn_path, mode="synonyms"):
    """Spoil sentences from queue_in and put them to queue_out."""
    exp = Experiment(wn_path)
    try:
        item = 1
        while item is not None:
            item = queue_in.get()  # InputSentence
            if item is not None:
                input_sentence = item
                if mode == "synonyms" or mode == "words":
                    substitutions = exp.get_substitutions(
                        input_sentence.lemmas,
                        input_sentence.synset_ids,
                        input_sentence.positions,
                        nbr_words,
                    )
                elif mode == "related":
                    substitutions = exp.get_related_unrelated(
                        input_sentence.lemmas,
                        input_sentence.synset_ids,
                        input_sentence.positions,
                        nbr_words,
                    )
                else:
                    raise ValueError("Wrong mode")
                queue_out.put(
                    (
                        input_sentence.sentence,
                        input_sentence.index,
                        substitutions,
                        input_sentence.orth,
                        mode,
                    )
                )

    except Exception as e:
        queue_log.put(f"Error in spoil: {e}")
        queue_in.put(None)
        queue_out.put(None)


def log_queues(queues):
    while True:
        sizes = [q.qsize() for q in queues]
        print(sizes, flush=True)
        sleep(10)


def log_info_queue(queue):
    print("Logging queue")
    while True:
        item = queue.get()
        if item is not None:
            print(item)
    print("Logging queue finished")


def load_dataset_list(name, lang):
    texts = []
    fn = f"{DATASET_LOCAL_PATH}/wsd_polish_datasets_{name}.jsonl" \
        if lang == "pl" else f"{DATASET_LOCAL_PATH}/wsd_english_datasets.jsonl"
    with open(
        fn, "r"
    ) as fin:
        for line in fin.readlines():
            if name.endswith("sentence"):
                sent = json.loads(line)["sentences"]
                texts.extend([dict(zip(sent, t)) for t in zip(*sent.values())])
            else:
                texts.append(json.loads(line))
    return texts


def experiment_run(
    dataset_name: str,
    emb_model: str,
    wn_path: str,
    spoilers_nbr: int = 2,
    nbr_words: int = 1,
    mode="synonyms",
    lang="pl",
):
    """Runs experiment.

    :param dataset_name: dataset name
    :param emb_model: model name for embeddings
    :param emb_pkl: path to file with precalculated org sentence embeddings
    :param wn_path: path to wordnet sqlite db file
    :param spoilers_nbr: number of spoiler processes
    :param nbr_words: number of words to change at once
    :param mode: mode to runs
    """
    print(f"Counting {emb_model} on {dataset_name}, {nbr_words} words changed")
    lines = load_dataset_list(dataset_name, lang)
    emb_pkl = f"{EMBBEDINGS_PATH}/embeddings_{emb_model}_{dataset_name}.pt"
    m = Manager()
    queues = [m.Queue(maxsize=QUEUE_SIZE) for _ in range(3)]
    queues.append(m.Queue(maxsize=max(int(1.5 * len(lines)), QUEUE_SIZE)))
    queues.append(m.Queue(maxsize=QUEUE_SIZE))

    log_queue_nbr = 4
    output_file = f"{RESULTS_PATH}/nlp_experiment_{emb_model}_{dataset_name}_{nbr_words}_{mode}.jsonl"

    log_que = Thread(target=log_queues, args=(queues[:3],))
    log_que.daemon = True
    log_que.start()
    info_que = Thread(target=log_info_queue, args=(queues[log_queue_nbr],))
    info_que.daemon = True
    info_que.start()

    processes = [
        Process(target=data_producer, args=(queues[0], lines, queues[log_queue_nbr], lang))
    ]

    processes.extend(
        [
            Process(
                target=spoil_queue,
                args=(
                    queues[0],
                    queues[1],
                    queues[log_queue_nbr],
                    nbr_words,
                    wn_path,
                    mode,
                ),
            )
            for _ in range(spoilers_nbr)
        ]
    )  # spoiling 0 -> 1

    processes.extend(
        [
            Process(
                target=count_similarity,
                args=(queues[1], queues[2], queues[log_queue_nbr], emb_model, emb_pkl, lang),
            ),
            Process(
                target=data_saver,
                args=(
                    queues[2],
                    queues[log_queue_nbr],
                    output_file,
                    len(lines),
                    queues[:3],
                ),
            ),
        ]
    )

    [p.start() for p in processes]

    # wait for all processes to finish
    [p.join() for p in processes]
    log_que.join(timeout=0.5)
    info_que.join(timeout=0.5)


def test_emb(model_name, dataset="wsd_english_datasets", lang="en"):
    model, tokenizer = load_model(model_name, lang)
    model_type = EmbeddingModelType(model_name)
    model = EmbeddingModel(model, tokenizer, model_type=model_type)
    lines = load_dataset_list(dataset, lang)
    line = lines[0]
    print(line)
    start_time = time.time()
    search_key = "sentence" if lang == "en" else "text"

    emb = model.get_embeddings(line[search_key])
    print(f"Time1 {time.time() - start_time}")
    line = lines[1]
    start_time = time.time()
    emb = model.get_embeddings(line[search_key])
    print(f"Time2 {time.time() - start_time}")
    print(emb.shape)
    print(emb)
    
    start_time = time.time()
    print(line.keys(), line)
    if lang == "en":
        emb = model.get_words_embeddings(line[search_key], line['orth'])
    else:
        emb = model.get_words_embeddings(line[search_key], line['tokens']['orth'])
    print(f"Time word {time.time() - start_time}")
    print(emb.shape)
    print(emb)


def count_emb(model_name, dataset="wsd_english_datasets", lang="en"):
    model, tokenizer = load_model(model_name, lang)
    model_type = EmbeddingModelType(model_name)
    model = EmbeddingModel(model, tokenizer, model_type=model_type)
    lines = load_dataset_list(dataset, lang)
    search_key = "sentence" if lang == "en" else "text"
    emb_pkl = f"{EMBBEDINGS_PATH}/embeddings_{model_name}_{dataset}.pt"
    emb_array = []
    for line in tqdm(lines):
        emb = model.get_embeddings(line[search_key])
        emb_array.append(emb)
    torch.save(torch.stack(emb_array), emb_pkl)



# EXAMPLE RUN
models = ["llama-hf"]
datasets = ["skladnica_sentence", "walenty_sentence"]
modes = ["related"]
WORDS_NUM = 1
for mode in modes:
    for dataset in datasets:
        for model in models:
            spoiler_nbr = 15
            print(f"Running {model} on {dataset} mode {mode} spoiler nbr {spoiler_nbr}")
            experiment_run(
                dataset,
                model,
                PLWN_API_PATH,
                spoilers_nbr=spoiler_nbr,
                nbr_words=WORDS_NUM,
                mode=mode,
                lang="pl"
            )

