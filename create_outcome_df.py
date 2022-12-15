import json
import os

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string


def remove_stopwords(text, language):
    stpword = stopwords.words(language)
    no_punctuation = [char for char in text if char not in string.punctuation]
    no_punctuation = ''.join(no_punctuation)
    return ' '.join([word for word in no_punctuation.split() if word.lower() not in stpword])


def create_df(path, crossvalfolds={}):
    arr = os.listdir(path)

    data = []

    for filename in arr:
    
        # crossval fold selection
        if filename.split('_')[1].split('.')[0] in crossvalfolds.keys():
            split = int(crossvalfolds[filename.split('_')[1].split('.')[0]])
        else:
            continue

        to_open = path + "\\" + filename

        with open(to_open, 'r', encoding='utf8') as f:
            datastore = json.load(f, strict=False)

        # The plainText is already in the JSON file
        plain_text = datastore['document']['plainText']

        # To find all the requests inside the document, we look for the 'req' annotations
        # The 'requests' dictionary will contain the annotation id as the key and a list with its start and end indicators
        requests = {}
        for annotation in datastore['annotations']:
            if annotation['name'] == 'req' and annotation['attributes']['G'] == '2':
                requests[annotation['_id']] = [annotation['start'], annotation['end']]        

        # Getting the <claim> and <arg> relative to the request
        claims_ids = {}
        claims = {}
        for req in requests.keys():
            for annotation in datastore['annotations']:
                if annotation['name'] == 'claim':
                    if 'PRO' in annotation['attributes'] and not isinstance(annotation['attributes']['PRO'], list):
                        annotation['attributes']['PRO'] = [annotation['attributes']['PRO']]
                    if 'CON' in annotation['attributes'] and not isinstance(annotation['attributes']['CON'], list):
                        annotation['attributes']['PRO'] = [annotation['attributes']['CON']]
                    if (('PRO' in annotation['attributes'] and req in list(annotation['attributes']['PRO'])) or
                            ('CON' in annotation['attributes'] and req in list(annotation['attributes']['CON']))):
                        if req in claims:
                            claims[req].append([annotation['start'], annotation['end']])
                            claims_ids[req].append(annotation['_id'])
                        else:
                            claims[req] = [[annotation['start'], annotation['end']]]
                            claims_ids[req] = [annotation['_id']]

        args = {}
        for req, values in claims_ids.items():
            for claim in values:
                for annotation in datastore['annotations']:
                    if annotation['name'] == 'arg':
                        if 'PRO' in annotation['attributes'] and not isinstance(annotation['attributes']['PRO'], list):
                            annotation['attributes']['PRO'] = [annotation['attributes']['PRO']]
                        if 'CON' in annotation['attributes'] and not isinstance(annotation['attributes']['CON'], list):
                            annotation['attributes']['PRO'] = [annotation['attributes']['CON']]
                        if (('PRO' in annotation['attributes'] and claim in list(annotation['attributes']['PRO'])) or
                                ('CON' in annotation['attributes'] and claim in list(annotation['attributes']['CON']))):
                            if req in args:
                                args[req].append([annotation['start'], annotation['end']])
                            else:
                                args[req] = [[annotation['start'], annotation['end']]]

        outcomes = {}
        # Looking for all the decision ('dec') annotations in the file to match them with their respective request
        for annotation in datastore['annotations']:
            if annotation['name'] == 'dec':
                # The 'O' attribute stands for 'object', meaning the object of the decision
                if 'O' in annotation['attributes']:
                    # A decision can have multiple objects
                    # Once we find the request that it refers to, we append the outcome of the decision to the list in the 'requests' dictionary
                    if isinstance(annotation['attributes']['O'], list):
                        for req in annotation['attributes']['O']:
                            if req in requests.keys():
                                # check che non ci sia già una dec con outcome opposto
                                if req in outcomes.keys() and outcomes[req] in ['0', '1']:
                                    if annotation['attributes']['E'] in ['0', '1'] and annotation['attributes']['E'] != outcomes[req]:
                                        outcomes.pop(req)
                                        continue
                                else:
                                    outcomes[req] = annotation['attributes']['E']
                    elif annotation['attributes']['O']:
                        if annotation['attributes']['O'] in requests.keys():
                            if annotation['attributes']['O'] in outcomes.keys() and outcomes[annotation['attributes']['O']] in ['0', '1']:
                                if annotation['attributes']['E'] in ['0', '1'] and annotation['attributes']['E'] != outcomes[annotation['attributes']['O']]:
                                    outcomes.pop(annotation['attributes']['O'])
                                    continue
                            else:
                                outcomes[annotation['attributes']['O']] = annotation['attributes']['E']
                else:
                    print(annotation['_id'] + ' does not contain the object attribute')

        # Removing from requests the ones that do not have a known outcome or do not have at least an arg or a claim
        temp = {}
        for req in requests:
            if req in outcomes.keys() and (req in args or req in claims):
                temp[req] = requests[req]

        requests = temp

        # adding dec nd mot columns to valid requests
        decs = {}
        mots = {}
        mots_of_claims = {}
        grades = {}
        for req in requests.keys():
            for annotation in datastore['annotations']:
                if annotation['name'] == 'req' and annotation['_id'] == req:
                    grades[req] = annotation['attributes']['G']
                if 'O' in annotation['attributes']:
                    if not isinstance(annotation['attributes']['O'], list):
                        annotation['attributes']['O'] = [annotation['attributes']['O']]

                    if annotation['name'] == 'dec' and annotation['attributes']['E'] in ['0', '1'] and req in list(annotation['attributes']['O']):
                        if req in decs:
                            decs[req].append([annotation['start'], annotation['end']])
                        else:
                            decs[req] = [[annotation['start'], annotation['end']]]

                    elif annotation['name'] == 'mot':
                        if req in list(annotation['attributes']['O']):
                            if req in mots:
                                mots[req].append([annotation['start'], annotation['end']])
                            else:
                                mots[req] = [[annotation['start'], annotation['end']]]

                        else:
                            for elem in list(annotation['attributes']['O']):
                                if elem in claims_ids[req]:
                                    if req in mots_of_claims:
                                        mots_of_claims[req].append([annotation['start'], annotation['end']])
                                    else:
                                        mots_of_claims[req] = [[annotation['start'], annotation['end']]]
                                    break

        language = filename.split('_')[0]

        # Creating the df row for each valid request
        for req, value in requests.items():
            print(datastore["document"]["name"])
            print('grade: ' + grades[req] + ' outcome:' + outcomes[req])
            if outcomes[req] in ['0', '1']:
                req_text = plain_text[value[0]:value[1]]
                args_text = ''
                if req in args:
                    for arg in args[req]:
                        args_text += plain_text[arg[0]:arg[1]] + ' '
                claims_text = ''
                if req in claims:
                    for claim in claims[req]:
                        claims_text += plain_text[claim[0]:claim[1]] + ' '
                decs_text = ''
                if req in decs:
                    for dec in decs[req]:
                        decs_text += plain_text[dec[0]:dec[1]] + ' '
                mots_text = ''
                if req in mots:
                    for mot in mots[req]:
                        mots_text += plain_text[mot[0]:mot[1]] + ' '

                motsofclaims_text = ''
                if req in mots_of_claims:
                    for motc in mots_of_claims[req]:
                        motsofclaims_text += plain_text[motc[0]:motc[1]] + ' '
                        
                data.append([remove_stopwords(req_text, language),
                             remove_stopwords(args_text, language),
                             remove_stopwords(claims_text, language),
                             remove_stopwords(mots_text, language),
                             remove_stopwords(decs_text, language),
                             remove_stopwords(motsofclaims_text, language),
                             grades[req], outcomes[req], split, filename])
          
    columns = ['request', 'args', 'claims', 'mots', 'decs', 'mots_of_claims', 'grade', 'outcome', 'split', 'document']
    df = pd.DataFrame(data, columns=columns)

    df.to_pickle("./df_outcome_vat.pkl")
    

crossval_folders = {'1000': '1', '1001': '2', '1002': '3', '1003': '4', '1004': '5', '1005': '1', '1006': '2', '1007': '3', '1008': '4', '1009': '3', '1010': '1', '1011': '2', '1012': '3', '1013': '2', '1014': '5', '1015': '1', '1016': '2', '1017': '3', '1018': '4', '1019': '5', '1020': '1', '1021': '2', '1022': '1', '1023': '4', '1024': '5', '1025': '1', '1026': '2', '1027': '3', '1028': '4', '1029': '5', '1030': '1', '1031': '4', '1032': '3', '1033': '4', '1034': '5', '1035': '1', '1036': '2', '1037': '3', '1038': '5', '1039': '5', '1040': '1', '1041': '2', '1042': '3', '1043': '4', '1044': '5', '1045': '1', '1046': '2', '1047': '3', '1048': '4', '1049': '5', '1050': '1', '1051': '2', '1052': '3', '1053': '4', '1054': '5', '1055': '1', '1056': '2', '1057': '1', '1058': '4', '1059': '5', '1060': '1', '1061': '2', '1062': '3', '1063': '4', '1064': '5', '1065': '1', '1066': '3', '1067': '3', '1068': '4', '1069': '5', '1070': '1', '1071': '2', '1072': '3', '1073': '4', '1074': '5', '1075': '1', '1076': '2', '1077': '3', '1078': '5', '1079': '5', '1080': '1', '1081': '2', '1082': '3', '1083': '4', '1084': '5', '1085': '1', '1086': '2', '1087': '3', '1088': '4', '1089': '5', '1090': '1', '1091': '2', '1092': '3', '1093': '4', '1094': '5', '1095': '3', '1096': '2', '1097': '3', '1098': '4', '1099': '5', '1100': '1', '1101': '2', '1102': '3', '1103': '4', '1104': '5', '1105': '1', '1106': '2', '1107': '3', '1108': '5', '1109': '5', '1110': '1', '1111': '2', '1112': '3', '1113': '4', '1114': '5', '1115': '1', '1116': '2', '1117': '3', '1118': '4', '1119': '5', '1120': '1', '1121': '2', '1122': '3', '1123': '4', '1124': '5', '1125': '1', '1126': '2', '1127': '3', '1128': '4', '1129': '5', '1130': '1', '1131': '2', '1132': '3', '1133': '4', '1134': '5', '1135': '1', '1136': '2', '1137': '3', '1138': '2', '1139': '5', '1140': '1', '1141': '2', '1142': '3', '1143': '3', '1144': '5', '1145': '1', '1146': '2', '1147': '3', '1148': '4', '1149': '5', '1150': '1', '1151': '2', '1152': '3', '1153': '4', '1154': '5', '1155': '1', '1156': '2', '1157': '3', '1158': '4', '1159': '5', '1160': '1', '1161': '2', '1162': '3', '1163': '4', '1164': '5', '1165': '1', '1166': '2', '1167': '3', '1168': '4', '1169': '5', '1170': '1', '1171': '2', '1172': '3', '1173': '4', '1174': '5', '1175': '1', '1176': '2', '1177': '3', '1178': '4', '1179': '5', '1180': '1', '1181': '2', '1182': '3', '1183': '4', '1184': '4', '1185': '1', '1186': '2', '1187': '3', '1188': '4', '1189': '5', '1190': '1', '1191': '2', '1192': '3', '1193': '1', '1194': '4', '1195': '1', '1196': '2', '1197': '3', '1198': '4', '1199': '4', '1200': '1', '1201': '2', '1202': '3', '1203': '4', '1204': '5', '1205': '1', '1206': '2', '1207': '3', '1208': '4', '1209': '5', '1210': '1', '1211': '2', '1212': '3', '1213': '4', '1214': '4', '1215': '1', '1216': '2', '1217': '3', '1218': '4', '1219': '5', '1220': '1', '1221': '2', '1222': '3', '1223': '4', '1224': '5', '1225': '1'}

create_df('.\\italianVAT_dataset', crossval_folders)