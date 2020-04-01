import pandas as pd


def cleanDataScheme(buffer):
    df = pd.read_json(buffer, lines=True)
    listTmp = []
    for giftAmountDict in df['style']:
        dictTmp = giftAmountDict
        if isinstance(giftAmountDict, dict):
            dictTmp = {}
            for key, value in giftAmountDict.items():
                keyCleaned = key.replace(" ", "_").replace(":", "")
                if isinstance(value, str):
                    valueCleaned = int(value)
                else:
                    print("ERROR!!!")
                    valueCleaned = 0
                dictTmp[keyCleaned] = valueCleaned

        listTmp.append(dictTmp)
    df['style'] = pd.Series(listTmp)
    return df

