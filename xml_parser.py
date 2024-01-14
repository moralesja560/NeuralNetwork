import xmltodict
import csv
import pandas as pd
import sys

#open the file
fileptr = open("temp_store_xml/620493.xml","r")

#read xml content from the file
xml_content= fileptr.read()
#print("XML content is:")
#print(xml_content)

my_ordered_dict=xmltodict.parse(xml_content)
#print("Ordered Dictionary is:")
#print(my_ordered_dict)

#store all dictionaries

# Method 1
"""
df1 = pd.DataFrame(my_ordered_dict['EXPORT']['FKBearbeitungsGruppe'],list(range(len(my_ordered_dict['EXPORT']['FKBearbeitungsGruppe']))))
df2 = pd.DataFrame(my_ordered_dict['EXPORT']['FKBaum'],list(range(len(my_ordered_dict['EXPORT']['FKBaum']))))
df3 = pd.DataFrame(my_ordered_dict['EXPORT']['FKBearbeitungsEinheit'],list(range(len(my_ordered_dict['EXPORT']['FKBearbeitungsEinheit']))))
df4 = pd.DataFrame(my_ordered_dict['EXPORT']['FKWZZuordKonfig'],list(range(len(my_ordered_dict['EXPORT']['FKWZZuordKonfig']))))
df5 = pd.DataFrame(my_ordered_dict['EXPORT']['FKStiftDornWZ'],list(range(len(my_ordered_dict['EXPORT']['FKStiftDornWZ']))))
df1.to_csv('temp_store_xml/FKBearbeitungsGruppe.csv', index=False, header=True)
df2.to_csv('temp_store_xml/FKBaum.csv', index=False, header=True)
df3.to_csv('temp_store_xml/FKBearbeitungsEinheit.csv', index=False, header=True)
df4.to_csv('temp_store_xml/FKWZZuordKonfig.csv', index=False, header=True)
df5.to_csv('temp_store_xml/FKStiftDornWZ.csv', index=False, header=True)


df8 = pd.DataFrame(my_ordered_dict['EXPORT']['FKRille'],list(range(len(my_ordered_dict['EXPORT']['FKRille']))))
df16 = pd.DataFrame(my_ordered_dict['EXPORT']['FKTeil'],list(range(len(my_ordered_dict['EXPORT']['FKTeil']))))
df20 = pd.DataFrame(my_ordered_dict['EXPORT']['FKTeilGlobDaten'],list(range(len(my_ordered_dict['EXPORT']['FKTeilGlobDaten']))))
df21 = pd.DataFrame(my_ordered_dict['EXPORT']['FKEinstellungen'],list(range(len(my_ordered_dict['EXPORT']['FKEinstellungen']))))
df22 = pd.DataFrame(my_ordered_dict['EXPORT']['FKEinstellWert'],list(range(len(my_ordered_dict['EXPORT']['FKEinstellWert']))))
df8.to_csv('temp_store_xml/FKRille.csv', index=False, header=True)
df16.to_csv('temp_store_xml/FKTeil.csv', index=False, header=True)
df20.to_csv('temp_store_xml/FKTeilGlobDaten.csv', index=False, header=True)
df21.to_csv('temp_store_xml/FKEinstellungen.csv', index=False, header=True)
df22.to_csv('temp_store_xml/FKEinstellWert.csv', index=False, header=True)


for i in range(0,len(my_ordered_dict['EXPORT']['FKWZKatalog'])):
	df6 = pd.DataFrame(my_ordered_dict['EXPORT']['FKWZKatalog'][i],list(range(len(my_ordered_dict['EXPORT']['FKWZKatalog'][i]))))
	df6.to_csv(f'temp_store_xml/FKWZKatalog{i}.csv', index=False, header=True)
	
#sys.exit()

for i in range(0,len(my_ordered_dict['EXPORT']['FKWZKatalog'])):
	df7 = pd.DataFrame(my_ordered_dict['EXPORT']['FKWZKatalog'][i],list(range(len(my_ordered_dict['EXPORT']['FKWZKatalog'][i]))))
	df7.to_csv(f'temp_store_xml/FKWZKatalog{i}.csv', index=False, header=True)
	

for i in range(0,len(my_ordered_dict['EXPORT']['FKRillenElement'])):
	df9 = pd.DataFrame(my_ordered_dict['EXPORT']['FKRillenElement'][i],list(range(len(my_ordered_dict['EXPORT']['FKRillenElement'][i]))))
	df9.to_csv(f'temp_store_xml/FKRillenElement{i}.csv', index=False, header=True)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKMaterial'])):
	df10 = pd.DataFrame(my_ordered_dict['EXPORT']['FKMaterial'][i],list(range(len(my_ordered_dict['EXPORT']['FKMaterial'][i]))))
	df10.to_csv(f'temp_store_xml/FKMaterial{i}.csv', index=False, header=True)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKMatKatalog'])):
	df11 = pd.DataFrame(my_ordered_dict['EXPORT']['FKMatKatalog'][i],list(range(len(my_ordered_dict['EXPORT']['FKMatKatalog'][i]))))
	df11.to_csv(f'temp_store_xml/FKMatKatalog{i}.csv', index=False, header=True)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKKennlinie'])):
	df12 = pd.DataFrame(my_ordered_dict['EXPORT']['FKKennlinie'][i],list(range(len(my_ordered_dict['EXPORT']['FKKennlinie'][i]))))
	df12.to_csv(f'temp_store_xml/FKKennlinie{i}.csv', index=False, header=True)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKKLKatalog'])):
	df13 = pd.DataFrame(my_ordered_dict['EXPORT']['FKKLKatalog'][i],list(range(len(my_ordered_dict['EXPORT']['FKKLKatalog'][i]))))
	df13.to_csv(f'temp_store_xml/FKKLKatalog{i}.csv', index=False, header=True)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKWZAblaufdaten'])):
	df14 = pd.DataFrame(my_ordered_dict['EXPORT']['FKWZAblaufdaten'][i],list(range(len(my_ordered_dict['EXPORT']['FKWZAblaufdaten'][i]))))
	df14.to_csv(f'temp_store_xml/FKWZAblaufdaten{i}.csv', index=False, header=True)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKWZAblaufdatenKatalog'])):
	df15 = pd.DataFrame(my_ordered_dict['EXPORT']['FKWZAblaufdatenKatalog'][i],list(range(len(my_ordered_dict['EXPORT']['FKWZAblaufdatenKatalog'][i]))))
	df15.to_csv(f'temp_store_xml/FKWZAblaufdatenKatalog{i}.csv', index=False, header=True)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKTeileKatalog'])):
	df17 = pd.DataFrame(my_ordered_dict['EXPORT']['FKTeileKatalog'][i],list(range(len(my_ordered_dict['EXPORT']['FKTeileKatalog'][i]))))
	df17.to_csv(f'temp_store_xml/FKTeileKatalog{i}.csv', index=False, header=True)
"""
for i in range(0,len(my_ordered_dict['EXPORT']['FKGeoElement'])):
	df18 = pd.DataFrame(my_ordered_dict['EXPORT']['FKGeoElement'][i],list(range(len(my_ordered_dict['EXPORT']['FKGeoElement'][i]))))
	df18.to_csv(f'temp_store_xml/FKGeoElement{i}.csv', index=False, header=True)

for i in range(0,len(my_ordered_dict['EXPORT']['FKMakroZeile'])):
	df19 = pd.DataFrame(my_ordered_dict['EXPORT']['FKMakroZeile'][i],list(range(len(my_ordered_dict['EXPORT']['FKMakroZeile'][i]))))
	df19.to_csv(f'temp_store_xml/FKMakroZeile{i}.csv', index=False, header=True)
	
"""
for i in range(0,len(my_ordered_dict['EXPORT']['FKGlobWert'])):
	df23 = pd.DataFrame(my_ordered_dict['EXPORT']['FKGlobWert'][i],list(range(len(my_ordered_dict['EXPORT']['FKGlobWert'][i]))))
	df23.to_csv(f'temp_store_xml/FKGlobWert{i}.csv', index=False, header=True)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKFreiesElement'])):
	df24 = pd.DataFrame(my_ordered_dict['EXPORT']['FKFreiesElement'][i],list(range(len(my_ordered_dict['EXPORT']['FKFreiesElement'][i]))))
	df24.to_csv(f'temp_store_xml/FKFreiesElement{i}.csv', index=False, header=True)
	
for i in range(0,len(my_ordered_dict['EXPORT']['FKKoordWert'])):
	df25 = pd.DataFrame(my_ordered_dict['EXPORT']['FKKoordWert'][i],list(range(len(my_ordered_dict['EXPORT']['FKKoordWert'][i]))))
	df25.to_csv(f'temp_store_xml/FKFreiesElement{i}.csv', index=False, header=True)
"""	




















