import xmltodict
import csv
import pandas as pd
import sys

#open the file
fileptr = open("620491.xml","r")


#read xml content from the file
xml_content= fileptr.read()
#print("XML content is:")
#print(xml_content)

my_ordered_dict=xmltodict.parse(xml_content)


step_number = []
D_pos = []
PAS_pos =[]
n_pos = []
n_acc_pos = []
HDorn_pos = []
va_pos = []


"""
for i in range(0,len(my_ordered_dict['EXPORT']['FKGeoElement'])):	
	
	df18 = pd.DataFrame(my_ordered_dict['EXPORT']['FKGeoElement'][i],list(range(len(my_ordered_dict['EXPORT']['FKGeoElement'][i]))))	
	
	df18.to_csv(f'temp_store_xml/FKGeoElement{i}.csv', index=False, header=True)
"""

#obtain the total steps
for i in range(0,len(my_ordered_dict['EXPORT']['FKGeoElement'])):	
	df18 = pd.DataFrame(my_ordered_dict['EXPORT']['FKGeoElement'][i],list(range(len(my_ordered_dict['EXPORT']['FKGeoElement'][i]))))	
	#Steps
	step_number.append(float(df18['NR'].iloc[0]))
	
	D_string = df18['GEO_WINDEDM'].iloc[0]
	temp_pos = D_string.find('|')
	if temp_pos > 0:
		D_pos.append(float(D_string[:temp_pos]))
	else:
		D_pos.append(df18['GEO_WINDEDM'].iloc[0])

	D_string = df18['GEO_STG'].iloc[0]
	temp_pos = D_string.find('|')
	if temp_pos > 0:
		PAS_pos.append(float(D_string[:temp_pos]))
	else:
		PAS_pos.append(df18['GEO_STG'].iloc[0])


	D_string = df18['GEO_WND'].iloc[0]
	temp_pos = D_string.find('|')
	if temp_pos > 0:
		n_pos.append(float(D_string[:temp_pos]))
	else:
		n_pos.append(df18['GEO_WND'].iloc[0])


	D_string = df18['GEO_GESWND'].iloc[0]
	temp_pos = D_string.find('|')
	if temp_pos > 0:
		n_acc_pos.append(float(D_string[:temp_pos]))
	else:
		n_acc_pos.append(df18['GEO_GESWND'].iloc[0])

	D_string = df18['TEC_DORNHOEHE'].iloc[0]
	temp_pos = D_string.find('|')
	if temp_pos > 0:
		HDorn_pos.append(float(D_string[:temp_pos]))
	else:
		HDorn_pos.append(df18['TEC_DORNHOEHE'].iloc[0])

	D_string = df18['TEC_GESCHW'].iloc[0]
	temp_pos = D_string.find('|')
	if temp_pos > 0:
		va_pos.append(float(D_string[:temp_pos]))
	else:
		va_pos.append(df18['TEC_GESCHW'].iloc[0])

Wafios_program = pd.DataFrame(
    {'step_number': step_number,
     'D_pos': D_pos,
     'PAS_pos': PAS_pos,
     'n_pos': n_pos,
     'n_acc_pos': n_acc_pos,
     'HDorn_pos': HDorn_pos,
     'va_pos': va_pos
    })
Wafios_program = Wafios_program.sort_values(by=['step_number'], ascending=True)

print(Wafios_program.head())

Wafios_program.to_csv("620491_processed.csv",index=False)


df16 = pd.DataFrame(my_ordered_dict['EXPORT']['FKTeil'],list(range(len(my_ordered_dict['EXPORT']['FKTeil']))))

with open('620493_processed.csv','a') as fd:
	part_number = df16['BESCHREIBUNG'].iloc[0]
	machine_number = df16['MASCHINENNUMMER'].iloc[0]
	comment = df16['KOMMENTAR'].iloc[0]
	fd.write(f"{part_number}-{comment} run in {machine_number}")

	