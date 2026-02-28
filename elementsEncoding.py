import pandas as pd

# Define the elements and their corresponding encoding values
elementsCommaEncoding = [('H', 1312), ('He', 2372), ('Li', 520.3), ('Be', 899.5), ('B', 800.7), # element, <value-to-be-embedded> 
                       ('C', 1086), ('N', 1402), ('O', 1314), ('F', 1681), ('Ne', 2081),
                       ('Na', 495.9), ('Mg', 737.8), ('Al', 577.6), ('Si', 786.5), ('P', 1012),
                       ('S', 999.6), ('Cl', 1251), ('Ar', 1521), ('K', 418.9), ('Ca', 589.8),
                       ('Sc', 631), ('Ti', 658), ('V', 650), ('Cr', 652.9), ('Mn', 717.4),
                       ('Fe', 759.4), ('Co', 758), ('Ni', 736.7), ('Cu', 745.5), ('Zn', 906.4),
                       ('Ga', 578.8), ('Ge', 762.2), ('As', 947), ('Se', 941), ('Br', 1140),
                       ('Kr', 1351), ('Rb', 403), ('Sr', 549.5), ('Y', 616), ('Zr', 660),
                       ('Nb', 664), ('Mo', 685), ('Tc', 702), ('Ru', 711), ('Rh', 720),
                       ('Pd', 805), ('Ag', 731), ('Cd', 867.7), ('In', 558.3), ('Sn', 708.6),
                       ('Sb', 833.8), ('Te', 869.3), ('I', 1008), ('Xe', 1170), ('Cs', 375.7),
                       ('Ba', 502.9), ('La', 538.1), ('Ce', 527.4), ('Pr', 523.2), ('Nd', 529.6),
                       ('Pm', 535.9), ('Sm', 543.3), ('Eu', 546.7), ('Gd', 592.6), ('Tb', 564.7),
                       ('Dy', 571.9), ('Ho', 580.7), ('Er', 588.7), ('Tm', 596.7), ('Yb', 603.4),
                       ('Lu', 523.6), ('Hf', 680), ('Ta', 761), ('W', 770), ('Re', 760),
                       ('Os', 840), ('Ir', 880)]

df = pd.read_csv('data1.csv')

dfs_to_concat = []
for index, row in df.iterrows():
    entry = str(row['Formula'])
    encoding_values = {}
    if isinstance(entry, str):
        i = 0
        while i < len(entry):
            if entry[i].isupper():
                element = entry[i]
                if i + 1 < len(entry) and entry[i + 1].islower():
                    element += entry[i + 1]
                    i += 1  # Move to the next character
                for elem, encoding in elementsCommaEncoding:
                    if elem == element:
                        encoding_values[elem] = encoding
                        break
            i += 1

    result_df = pd.DataFrame({'Formula_dcg': [entry], **encoding_values})
    dfs_to_concat.append(result_df)


final_result_df = pd.concat(dfs_to_concat, ignore_index=True)
final_result_df.to_csv('element_encoded_values.csv', index=False)

