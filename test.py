from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy.lang.ru import Russian
import pymorphy2
import json
import io
from spacy import displacy
#from spacy.gold import biluo_tags_from_offsets
#dataExample

with io.open('C:\pip\Main\data1.json', encoding='utf8') as f1:
    TRAIN_DATA = json.load(f1)

with io.open('C:\pip\Main\data2.json', encoding='utf8') as f2:
    TEST_DATA = json.load(f2)
	
#TRAIN_DATA = [
 #       ("Представитель Минобороны Мали сообщил о нападении боевиков на патруль вооруженных сил на востоке страны.", {"entities": [(25, 29, "PERSON")]}),
  #      ("Об этом сообщили ТАСС в аэропорту Южно-Сахалинска.", {"entities": [(17, 21, "ORG"), (34, 49, "LOC")]}),
   #     ("Занимавший эту должность Стивен Ло отправлен в отставку.", {"entities": [(25, 34, "PERSON")]})]

#TEST_DATA = [       
 #       ("Ранее Госсекретарь США Майк Помпео рассказал о разнице в признании одностороннего провозглашения Израилем суверенитета над Голанскими высотами и осуждением присоединения Крыма к России.", {"entities": [()]}
  #       )]
        
@plac.annotations(
    model=("Model name. Defaults to blank 'ru' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=100):
    if model is not None:
        nlp = spacy.load(model)  # загрузить существующую модель spaCy
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("ru")  # создать пустой языковой класс
        print("Created blank 'ru' model")

    # создать встроенные компоненты конвейера и добавить их в конвейер
    # nlp.create_pipe работает для встроенных модулей, которые зарегистрированы в spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # в противном случае, получите это, чтобы мы могли добавить ярлыки (labels)
    else:
        ner = nlp.get_pipe("ner")

    # добавить ярлыки (labels)
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # получить имена других каналов, чтобы отключить их во время тренировки
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # сбросить и инициализировать веса случайным образом - но только если мы
        # обучаем новую модель
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # сгруппировать примеры, используя мини-пакет spaCy
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # пакет текстов
                    annotations,  # пакет аннотаций
                    drop=0.5,  # выпадение - усложнить запоминание данных
                    losses=losses,
                )
            print("Losses", losses)

    	# проверить обученную модель
    for text, _ in TEST_DATA:
        doc = nlp(text)
		#displacy.serve(doc, style="ent") # visualizer
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    #Запись в файл output_data размеченных тестовых данных ( при помощи тренировочных )
    with io.open('C:\pip\Main\output_data.json', 'w', encoding='utf8') as f3:
        for text, _ in TEST_DATA:
        	doc = nlp(text)
        	json.dump([(t.text, t.ent_type_, text.find(t.text), text.find(t.text)+len(t.text)) for t in doc], f3, ensure_ascii=False)

    # сохранить модель в выходной каталог
#    if output_dir is not None:
#        output_dir = Path(output_dir)
#        if not output_dir.exists():
#            output_dir.mkdir()
#        nlp.to_disk(output_dir)
#        print("Saved model to", output_dir)

        # проверить сохраненную модель
 #       print("Loading from", output_dir)
 #       nlp2 = spacy.load(output_dir)
 #       for text, _ in TRAIN_DATA:
 #           doc = nlp2(text)
 #           print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
 #           print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

#if __name__ == "__main__":
    #plac.call(main)
main()