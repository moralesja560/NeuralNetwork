import xmltodict
import csv
import pandas as pd
import sys

file_part_number = '620493'

#open the file
fileptr = open(f"temp_store_xml/{file_part_number}.xml","r")


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
Ncode_ID3 = []
NC_satz = []


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

for i in range(0,len(my_ordered_dict['EXPORT']['FKMakroZeile'])):	
	df19 = pd.DataFrame(my_ordered_dict['EXPORT']['FKMakroZeile'][i],list(range(len(my_ordered_dict['EXPORT']['FKMakroZeile'][i]))))	
	
	Ncode_ID3.append(df19['DBBAUM_ID'].iloc[0])

	NC_satz.append(df19['NCSATZ'].iloc[0])




Wafios_program = pd.DataFrame(
    {'step_number': step_number,
     'D_pos': D_pos,
     'PAS_pos': PAS_pos,
     'n_pos': n_pos,
     'n_acc_pos': n_acc_pos,
     'HDorn_pos': HDorn_pos,
     'va_pos': va_pos
    })

Wafios_NC = pd.DataFrame(
    {'Ncode_ID3': Ncode_ID3,
     'NC_satz': NC_satz
    })

Wafios_program = Wafios_program.sort_values(by=['step_number'], ascending=True)


#print(Wafios_program.head())

Wafios_program.to_csv(f"temp_store_xml/{file_part_number}_processed.csv",index=False)
Wafios_NC.to_csv(f"temp_store_xml/{file_part_number}_NC_processed.csv",index=False)


#Agregar datos de maquina.

df16 = pd.DataFrame(my_ordered_dict['EXPORT']['FKTeil'],list(range(len(my_ordered_dict['EXPORT']['FKTeil']))))

with open(f'temp_store_xml/{file_part_number}_processed.csv','a') as fd:
	try:
		part_number = df16['BESCHREIBUNG'].iloc[0]
	except:
		part_number = "NA"
	try:
		machine_number = df16['MASCHINENNUMMER'].iloc[0]
	except:
		machine_number = "NA"
	try:
		comment = df16['KOMMENTAR'].iloc[0]
	except:
		comment = "NA"
	fd.write(f"{part_number}-{comment} run in {machine_number}")

	