from flair.models import TARSTagger
from flair.data import Sentence
import os, json
import spacy
from spacy.lang.en import English
from pathlib import Path
import rubrix as rb
from tqdm import tqdm

"""
This processes the files in the following steps
1. Extract sentences using Spacy
2. ZeroShot NER using Flair's TARS
3. Export the results in a json file that can be future process by ingesting into Rubrix or convert to BIO
"""

def process_file(tars, nlp, filename):
    """
    Process the filename and apply spacy and TARS NER to extract the entities
    :param tagger:
    :param nlp: Spacy NLP
    :param filename:
    :return: a list of tuples (Flair Sentence obj, sentence text)
    """
    try:
        data = Path(filename).read_text("utf8")
        data = data.replace("\n", " ")
        doc = nlp(data)
        # We need to keep the Spacy Sentences because the Flair Sentence object
        # mangles the text inserting spaces before puntuation
        sentences = [(Sentence(i.text), i.text) for i in doc.sents]
        flair_sentence = [s[0] for s in sentences]
        '''
        for sentence in sentences:
            tars.predict(sentence[0])
            print(sentence[0].to_tagged_string("ner"))
        '''
        tars.predict(flair_sentence)
        #print(flair_sentence)
        return sentences
    except FileNotFoundError:
        print(filename, "not found")
        #os.rename(filename, os.path.join(error_dir, Path(filename).name))
        return []


def create_json(sentences):
    records = []
    for s in sentences:
        prediction = [
            (entity.get_labels()[0].value, entity.start_position, entity.end_position)
            for entity in s[0].get_spans("ner")
        ]

        #print(s[0].text)
        #print([token.text for token in s[0]])
        #print(prediction)
        # We have the use the sentence that is from Spacy cause the Flair sentence introduced unnecessary spaces inside
        # u can compate s[0].text vs s[1] to see what i mean
        r = {"text": s[1],
             "tokens": [token.text for token in s[0]],
             "predictions": prediction}

        records.append(r)
    return records


def create_token_classification_records(sentences):
    records = []
    for s in sentences:
        prediction = [
            (entity.get_labels()[0].value, entity.start_position, entity.end_position)
            for entity in s[0].get_spans("ner")
        ]

        #print(s[0].text)
        #print([token.text for token in s[0]])
        #print(prediction)
        # We have the use the sentence that is from Spacy cause the Flair sentence introduced unnecessary spaces inside
        # u can compate s[0].text vs s[1] to see what i mean
        r = rb.TokenClassificationRecord(text=s[1],
                                         tokens=[token.text for token in s[0]],
                                         prediction=prediction)
        records.append(r)
    return records


def process_directory(root_dir, tars, nlp):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in tqdm(filenames):
            path_obj = Path(os.path.join("./json", f + ".json"))
            if not path_obj.exists():
                sen = process_file(tars, nlp, os.path.join(root_dir, f))
                j = create_json(sen)
                if len(j) != 0:
                    with path_obj.open('w', encoding="utf8") as fp:
                        json.dump(j, fp,
                                indent=2, sort_keys=True)


if __name__ == "__main__":
    # 1. Load zero-shot NER tagger
    tars = TARSTagger.load('tars-ner')

    # 3. Define some classes of named entities such as "soccer teams", "TV shows" and "rivers"
    labels = ["Aircraft", "Ship", "Boat"]
    tars.add_and_switch_to_new_task('task 1', labels, label_type='ner')

    nlp = spacy.load("en_core_web_sm")
    list_files = [r"D:\workspace\nerFlairTagger\data\borneobulletin_com_bn_canadian-frigate-arrives-in-sultanate_"]
    '''
    for filepath in list_files:
        sen = process_file(tars, nlp, filepath)
        # records = create_token_classification_records(sen)
        # rb.log(records, name="news3")

        j = create_json(sen)

        print(Path(filepath).name)
        with Path(os.path.join("./json", Path(filepath).name+".json")).open('w', encoding="utf8") as f:
            json.dump(j, f,
                      indent=2, sort_keys=True)
    '''

    process_directory("./data", tars, nlp)

    ### Below is the code to change into the BIO format
    """
    new_predictions = []
    prediction_index = 0
    if len(prediction) != 0:
        predict = ""
        for token in s.tokens:
            label = prediction[prediction_index][0]
            start_pos = prediction[prediction_index][1]
            end_pos = prediction[prediction_index][2]
            if token.start_pos == start_pos:
                predict = "B-"+label
            elif token.end_pos == end_pos:
                predict = "I-" + label
                if prediction_index+1 != len(prediction):
                    prediction_index += 1
            elif token.start_pos >= start_pos and token.end_pos <= end_pos:
                predict = "I-" + label
            else:
                predict = "O"
            new_predictions.append((predict, token.start_pos, token.end_pos))
            # building TokenClassificationRecord

            records.append(
                rb.TokenClassificationRecord(
                    text=s.text,
                    tokens=[token.text for token in s],
                    prediction=new_predictions,
                    prediction_agent="tars-ner",
                )
            )
        """

