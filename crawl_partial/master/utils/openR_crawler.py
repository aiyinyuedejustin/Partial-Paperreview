import openreview
import time
import requests
import jsonlines


class OpenreviewCrawler:
    def __init__(self, baseurl='https://api.openreview.net'):
        """need代理"""
        self.client = openreview.Client(baseurl=baseurl)
        self.venues = self.client.get_group(id='venues').members

    def get_and_save_venue(self, venue_id):
        results_list = self._get_venue_papers(venue_id)
        if results_list:
            self._save_results(results_list)
        return results_list

    def get_and_save_total(self):
        total_results_list = []
        for idx, venue_id in enumerate(self.venues):
            print('{}/{}: {}, total_results_list_length: {}'.format(idx + 1, len(self.venues), venue_id,
                                                                    len(total_results_list)))

            results_list = self._get_venue_papers(venue_id)
            total_results_list += results_list
            time.sleep(1)
        self._save_results(total_results_list, spec_name='total_notes')
        print('The number of papers is {}.'.format(len(total_results_list)))
        return total_results_list

    def _get_venue_papers(self, venue_id):
        """
        从venues（venues=client.get_group(id='venues').members）中获取指定venue的id来传入，
        该函数将返回对应venue_id的论文信息并存储
        """
        #         assert self._existence_check(venue_id), \
        #             'This item "{}" is not available in openviewer.net!'.format(venue_id)
        # 获取当前venue_id对应的提交论文（双盲）
        submissions = self.client.get_all_notes(invitation='{}/-/Blind_Submission'.format(venue_id),
                                                details='directReplies')

        # 获取当前venue_id下的论文id
        specified_forum_ids = self._get_all_forum_ids(submissions)

        # dict list
        results_list = [self._format_note(note, venue_id)
                        for note in submissions
                        if note.forum in specified_forum_ids]

        #         if results_list:
        #             for i in range(3):
        #                 print(results_list[i]['basic_dict']['forum'])
        return results_list

    def _get_specified_forum_ids(self, submissions):
        forum_ids = set()
        for note in submissions:
            for reply in note.details["directReplies"]:
                forum_ids.add(reply['forum'])
        return forum_ids

    def _get_all_forum_ids(self, submissions):
        """获取所有论文页id，无论是否有reply"""
        forum_ids = set()
        for note in submissions:
            forum_ids.add(note.forum)
        return forum_ids

    def _format_note(self, note, venue_id):
        """单条note的处理方法：提取note中的指定信息"""
        basic_dict = {}
        reviews_msg = []

        authors_string = ','.join(note.content.get('authors', '--'))
        keywords_string = ','.join(note.content.get('keywords', '--'))

        localtime_string = time.strftime('%Y-%m-%d', time.localtime(note.pdate / 1000)) if note.pdate else '--'

        # basic message
        basic_dict['forum'] = note.forum if note.forum else '--'
        basic_dict['title'] = note.content.get('title', '--')
        basic_dict['url'] = 'https://openreview.net/forum?id=' + note.forum
        basic_dict['pub_date'] = localtime_string
        basic_dict['abstract'] = note.content.get('abstract', '--')
        basic_dict['TL;DR'] = note.content.get('TL;DR', '--')
        basic_dict['authors'] = authors_string
        basic_dict['keywords'] = keywords_string
        basic_dict['venue'] = note.content.get('venue', '--')
        basic_dict['venue_id'] = note.content.get('venueid', '--')
        basic_dict['number'] = note.number if note.number else '--'
        basic_dict['pdf_url'] = 'https://openreview.net/pdf?id=' + note.forum
        basic_dict['signatures'] = note.signatures if note.signatures else '--'
        basic_dict['bibtex'] = note.content.get('_bibtex', '--')
        basic_dict['from_venue_id'] = venue_id

        # reviews message
        reviews_msg = note.details["directReplies"]

        result_dict = {'basic_dict': basic_dict, 'reviews_msg': reviews_msg}

        return result_dict

    def _existence_check(self, item_id):
        if requests.get("https://openreview.net/group?id={}".format(item_id)).status_code == 200:
            return True
        else:
            return False

    def _save_results(self, results_list, spec_name=None):
        if spec_name:
            venue_id = spec_name
            jsonl_file_name = '{}.jsonl'.format(spec_name)
        else:
            venue_id = results_list[0]['basic_dict']['venue_id']
            jsonl_file_name = '{}.jsonl'.format(venue_id.replace(r'/', '--').replace(r'.', '__'))
        for result in results_list:
            with jsonlines.open(jsonl_file_name, mode='a') as file:
                file.write(result)
        print('The item "{}" saved successfully!'.format(venue_id))
        return


if __name__ == '__main__':
    orc = OpenreviewCrawler()
    results_list = orc.get_and_save_venue('ICLR.cc/2023/Workshop/TSRL4H')
    print(results_list[:3])