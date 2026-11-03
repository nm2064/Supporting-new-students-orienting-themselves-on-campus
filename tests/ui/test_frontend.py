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
    assert image.evaluate('(img) => img.complete && img.naturalWidth > 0')
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
