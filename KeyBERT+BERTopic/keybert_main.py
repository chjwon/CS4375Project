from keybert import KeyBERT



def get_KeyBert_result(document_list):
    result = []
    kw_model = KeyBERT()

    for doc in document_list:
        keywords = kw_model.extract_keywords(doc)
        if len(keywords) == 0:
            print("None in keyBert")
            result.append([" "])
        else:
            result.append(keywords)
    return result

def keywordSave(keywordList, dataName):
    with open('./{}_keyWordList.txt.'.format(dataName),'w') as kl:
        for kList in keywordList:
            for keyword in kList:
                kl.write("%s, " %keyword[0])
            kl.write("\n")