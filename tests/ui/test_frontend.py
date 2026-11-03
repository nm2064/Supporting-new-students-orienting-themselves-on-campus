"""Public browser seams. Run: python -m pytest tests/ui -q."""
import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import pytest
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture
def page():
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.split('?')[0] in ['/chat', '/map', '/services', '/events', '/onboarding']:
                self.path = '/index.html'
            super().do_GET()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(ROOT)))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width':390,'height':844}, device_scale_factor=1)
        page.set_default_timeout(7000)
        page.route('**/reset-chat', lambda r: r.fulfill(json={'status':'reset'}))
        page.base_url = f'http://127.0.0.1:{server.server_port}'
        yield page
        browser.close()
    server.shutdown(); server.server_close()

def test_home_has_campus_photography_and_clear_mobile_actions(page):
    page.goto(page.base_url)
    page.get_by_role('heading', level=2).first.wait_for()
    image = page.get_by_role('img', name='Edinburgh campus')
    assert image.count() == 1
    image.evaluate('(img) => img.decode()')
    assert image.evaluate('(img) => img.naturalWidth > 0')
    assert page.get_by_role('link', name='Explore campus').count() == 1
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')


def test_mobile_map_leaves_room_for_map_and_directions(page):
    page.route('**/api/places*', lambda r: r.fulfill(json=[]))
    page.route('**/api/opening-hours*', lambda r: r.fulfill(json=[]))
    page.goto(page.base_url + '/map')
    canvas = page.locator('.map-canvas')
    canvas.wait_for()
    box = canvas.bounding_box()
    assert box['width'] >= 350
    assert box['height'] >= 240
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')


def test_event_card_has_date_venue_and_source_action_on_mobile(page):
    page.route('**/api/events*', lambda r: r.fulfill(json=[{
        'title':'Tuesday Soup Lunch','start':'2026-11-03T12:30:00','end':'2026-11-03T14:00:00',
        'location':'Chaplaincy, Edinburgh Campus','url':'https://hw.ac.libcal.com/event/4567341',
    }]))
    page.goto(page.base_url + '/events')
    card = page.get_by_role('article', name='Tuesday Soup Lunch')
    card.wait_for()
    assert card.get_by_role('link', name='View event').get_attribute('href') == 'https://hw.ac.libcal.com/event/4567341'
    assert card.get_by_text('Chaplaincy, Edinburgh Campus').is_visible()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')


@pytest.mark.parametrize('width', [360, 768, 1440])
def test_home_layout_and_language_settings_work_across_sizes(page, width):
    page.set_viewport_size({'width':width,'height':900})
    page.goto(page.base_url)
    page.get_by_text('Settings', exact=True).click()
    page.get_by_label('Language', exact=True).select_option('ar')
    assert page.locator('html').get_attribute('dir') == 'rtl'
    page.get_by_label('Dark mode', exact=True).check()
    assert 'dark' in page.locator('html').get_attribute('class')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.reload()
    page.get_by_role('heading', level=2).first.wait_for()
    assert page.locator('html').get_attribute('dir') == 'rtl'


def test_chat_starter_card_fills_message_without_sending(page):
    page.goto(page.base_url + '/chat')
    page.get_by_role('button', name='Library opening hours', exact=True).click()
    assert page.get_by_role('textbox', name='Message to UniBot').input_value() == 'When is the library open?'


@pytest.mark.parametrize('width', [360, 1440])
def test_settings_opens_without_moving_header_or_content_and_escape_returns_focus(page, width):
    page.set_viewport_size({'width':width,'height':900})
    page.goto(page.base_url)
    page.get_by_role('heading', level=2).first.wait_for()
    page.evaluate('document.fonts.ready')
    header_before = page.locator('header').bounding_box()
    content_before = page.locator('main').bounding_box()
    page.get_by_text('Settings', exact=True).click()
    assert page.locator('header').bounding_box() == header_before
    assert page.locator('main').bounding_box() == content_before
    panel = page.get_by_role('dialog', name='Settings')
    panel.wait_for()
    box = panel.bounding_box()
    assert box['x'] >= 0 and box['x'] + box['width'] <= width
    if width > 640:
        trigger_box = page.get_by_role('button', name='Settings', exact=True).bounding_box()
        assert abs(box['x'] + box['width'] - trigger_box['x'] - trigger_box['width']) < 2
    page.keyboard.press('Escape')
    assert not panel.is_visible()
    assert page.get_by_role('button', name='Settings', exact=True).evaluate('(el) => el === document.activeElement')


def test_settings_done_and_outside_click_dismiss_panel_and_keep_preferences(page):
    page.goto(page.base_url)
    page.get_by_role('button', name='Settings', exact=True).click()
    panel = page.get_by_role('dialog', name='Settings')
    panel.get_by_label('Campus', exact=True).select_option('dubai')
    panel.get_by_role('button', name='Done', exact=True).click()
    assert not panel.is_visible()
    page.get_by_role('button', name='Settings', exact=True).click()
    assert panel.get_by_label('Campus', exact=True).input_value() == 'dubai'
    page.mouse.click(5, 700)
    assert not panel.is_visible()
    page.reload()
    page.get_by_role('button', name='Settings', exact=True).click()
    assert page.get_by_label('Campus', exact=True).input_value() == 'dubai'


def test_chinese_settings_and_home_have_localized_controls(page):
    page.goto(page.base_url)
    page.get_by_role('button', name='Settings', exact=True).click()
    page.get_by_label('Language', exact=True).select_option('zh-Hans')
    assert page.get_by_role('heading', name='偏好设置', exact=True).is_visible()
    assert page.get_by_role('button', name='完成', exact=True).is_visible()
    page.get_by_role('button', name='完成', exact=True).click()
    assert page.get_by_role('heading', name='开启你的校园生活', exact=True).is_visible()
    assert page.get_by_role('link', name='探索校园').is_visible()


def test_chinese_chat_starters_fill_chinese_messages(page):
    page.goto(page.base_url)
    page.get_by_role('button', name='Settings', exact=True).click()
    page.get_by_label('Language', exact=True).select_option('zh-Hans')
    page.get_by_role('button', name='完成', exact=True).click()
    page.goto(page.base_url + '/chat')
    page.get_by_role('button', name='图书馆开放时间', exact=True).click()
    assert page.get_by_role('textbox', name='发送给校园助手的消息').input_value() == '图书馆什么时候开放？'
    assert page.get_by_role('button', name='发送消息', exact=True).is_visible()
