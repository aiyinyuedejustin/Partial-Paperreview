import jsonlines
import pandas as pd


class OpenreviewProccessor:
    def __init__(self, jsonl_path):
        self.df = self._load_jsonl_to_dataframe(jsonl_path)
        self.df_sub = pd.DataFrame()

    def _load_jsonl_to_dataframe(self, jsonl_path):
        msg_list = []
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            for line_dict in jsonlines.Reader(file):
                msg_dict = {}
                for k, v in line_dict['basic_dict'].items():
                    msg_dict['b_' + k] = v
                msg_list.append(msg_dict)
                for review_msg in line_dict["reviews_msg"]:
                    msg_dict_copy = msg_dict.copy()
                    pure_review_msg = {
                        'r_id': review_msg.get('id', None),
                        'r_number': review_msg.get('number', None),
                        'r_replyto': review_msg.get('replyto', None),
                        'r_invitation': review_msg.get('invitation', None),
                        'r_signatures': ','.join(review_msg['signatures']) if review_msg.get('signatures', None) else None,
                        'r_readers': review_msg.get('readers', None),
                        'r_nonreaders': review_msg.get('nonreaders', None),
                        'r_writers': review_msg.get('writers', None)
                    }
                    pure_content_msg = {}
                    pure_content_msg['c_content'] = review_msg['content']
                    for k, v in review_msg['content'].items():
                        pure_content_msg['c_' + k] = v
                    pure_review_msg.update(pure_content_msg)
                    msg_dict_copy.update(pure_review_msg)
                    msg_list.append(msg_dict_copy)
        dataframe = pd.DataFrame(msg_list)
        dataframe['c_final_decision'] = self._fill_decision(dataframe)
        return dataframe

    def _fill_decision(self, dataframe):
        return dataframe['c_decision'].map(lambda x: x if pd.isnull(x) else
                                           'Accepted' if 'accept' in x.lower() else
                                            'Rejected' if 'reject' in x.lower() else "Unknown")

    def get_sub(self, mode=None):
        # 仅带有review的df
        df_sub = self.df.dropna(subset=self.df.filter(regex='^(?!b_*)').columns, how='all')
        if mode == 'decision':
            # review类型仅为decision的df
            df_sub = df_sub[df_sub['r_invitation'].str.contains('Decision')]
        elif mode == 'other':
            # review类型仅为非decision的df
            df_sub = df_sub[~df_sub['r_invitation'].str.contains('Decision')]
        elif mode == 'accepted':
            # decision中被采纳的df
            df_sub = df_sub[df_sub['c_final_decision'].isin(['Accepted'])]
        elif mode == 'rejected':
            # decision中未被采纳的df
            df_sub = df_sub[df_sub['c_final_decision'].isin(['Rejected'])]

        self.df_sub = df_sub
        return

    def get_total_shape(self):
        return self.df.shape

    def get_sub_shape(self):
        return self.df_sub.shape


if __name__ == '__main__':
    orp = OpenreviewProccessor('../total_notes.jsonl')
    orp.get_sub()
    print(orp.df_sub.iloc[0])
