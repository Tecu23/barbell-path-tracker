# imports
from redvid import Downloader


# Contants
OUTPUT_PATH = 'videos'


# extracts video to '/videos'
def extract_video(reddit_link):
    reddit = Downloader()
    reddit.max_s = 5 * (1 << 20) # 5 MB
    reddit.path = OUTPUT_PATH
    reddit.url = reddit_link

    reddit.download()


def main():
    videos_link = [
        'https://www.reddit.com/r/formcheck/comments/133dtne/deadlift_form_check/',
        'https://www.reddit.com/r/formcheck/comments/132nrxa/deadlift_form_check/',
        'https://www.reddit.com/r/formcheck/comments/132u97q/op/',
        'https://www.reddit.com/r/formcheck/comments/1329eaf/front_squat/',
        'https://www.reddit.com/r/formcheck/comments/12xybjc/3x205lbs_any_visible_issues/',
        'https://www.reddit.com/r/formcheck/comments/12sadnh/hi_there_i_am_17_just_wanted_a_check_on_form_and/',
        'https://www.reddit.com/r/formcheck/comments/12q8305/is_my_back_rounding_and_am_i_going_to_deep/',
        'https://www.reddit.com/r/formcheck/comments/12k4jt1/backsquat_165x3/',
        'https://www.reddit.com/r/formcheck/comments/12exiz0/form_check_90kg_last_set/',
        'https://www.reddit.com/r/formcheck/comments/12err55/squat_form/',
        'https://www.reddit.com/r/formcheck/comments/12c20yq/squat_check_extra_information_in_the_comments/',
        'https://www.reddit.com/r/formcheck/comments/1296csx/275x2_155lb_bw/',
        'https://www.reddit.com/r/formcheck/comments/126odc2/squat_pain_need_some_form_tips/',
        'https://www.reddit.com/r/formcheck/comments/122v0tp/rpe10_final_set_of_squats_with_100kg_is_my_form/',
        'https://www.reddit.com/r/formcheck/comments/131zjfx/sumo_form_check_please/',
        'https://www.reddit.com/r/formcheck/comments/12zb96c/appreciate_a_form_check_on_my_45kg_deadlift/',
        'https://www.reddit.com/r/formcheck/comments/12ypq1n/140kgx3_1rm_is_170kg/',
        'https://www.reddit.com/r/formcheck/comments/12yocx2/is_my_deadlift_any_good/',
        'https://www.reddit.com/r/formcheck/comments/12vtghe/need_help_is_my_back_rounding_is_my_hip_a_little/',
        'https://www.reddit.com/r/formcheck/comments/12lvhx0/how_can_i_get_my_butt_to_drop_directly_down/',
        ]


    for link in videos_link:
        extract_video(link)


if __name__ == "__main__":
    main()
