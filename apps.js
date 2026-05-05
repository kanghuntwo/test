// Created: 2026-05-05 18:04:02
window.APP_STORAGE_KEY = 'appList.v1';

window.APP_INITIAL = [
  { name: '카카오톡', info: '국내에서 가장 많이 사용되는 모바일 메신저 앱으로, 채팅·음성·영상통화 및 송금 기능을 제공합니다.' },
  { name: '네이버', info: '검색, 뉴스, 쇼핑, 지도 등 다양한 서비스를 통합 제공하는 포털 앱입니다.' },
  { name: '유튜브', info: '구글이 운영하는 세계 최대 동영상 공유 플랫폼으로, 영상 시청과 라이브 스트리밍을 지원합니다.' },
  { name: '인스타그램', info: '사진과 짧은 동영상을 공유하는 소셜 네트워크 서비스로, 스토리·릴스 기능을 제공합니다.' },
  { name: '쿠팡', info: '로켓배송으로 잘 알려진 국내 대표 이커머스 앱으로, 빠른 배송과 다양한 상품을 제공합니다.' },
  { name: '토스', info: '간편 송금에서 시작해 증권·보험·대출까지 통합 제공하는 종합 금융 플랫폼 앱입니다.' },
  { name: '배달의민족', info: '국내 최대 음식 배달 주문 앱으로, 다양한 음식점 정보와 빠른 주문·결제 기능을 제공합니다.' },
  { name: '페이스북', info: '메타가 운영하는 글로벌 소셜 네트워크 서비스로, 친구·가족과 게시물·사진을 공유합니다.' },
  { name: '엑스(X)', info: '구 트위터로, 짧은 글과 이미지로 실시간 소식과 의견을 공유하는 마이크로블로그 SNS입니다.' },
  { name: '틱톡', info: '짧은 길이의 동영상을 손쉽게 제작·공유하는 글로벌 숏폼 플랫폼으로, 다양한 트렌드를 만들어 냅니다.' },
  { name: '넷플릭스', info: '영화·드라마·다큐 등 다양한 콘텐츠를 스트리밍으로 제공하는 글로벌 OTT 서비스입니다.' },
  { name: '디스코드', info: '음성·텍스트·화상 채팅을 지원하는 커뮤니티 메신저로, 게이머와 다양한 그룹이 활용합니다.' },
  { name: '줌', info: '화상회의·웨비나·온라인 강의에 널리 사용되는 영상 회의 솔루션입니다.' },
  { name: '슬랙', info: '채널 기반 협업 메신저로, 팀 커뮤니케이션과 외부 도구 연동을 지원합니다.' },
  { name: '노션', info: '문서·노트·데이터베이스·위키를 한 곳에서 관리할 수 있는 올인원 워크스페이스 도구입니다.' },
  { name: '스포티파이', info: '전 세계에서 널리 사용되는 음악 스트리밍 서비스로, 개인화된 추천 플레이리스트를 제공합니다.' },
  { name: '멜론', info: '국내 최대 음원 스트리밍 서비스로, 최신 차트와 다양한 장르의 음악을 제공합니다.' },
  { name: '카카오맵', info: '국내 도로·대중교통·도보 길찾기를 지원하는 카카오의 지도·내비게이션 앱입니다.' },
  { name: '네이버 지도', info: '실시간 교통정보와 정확한 길찾기, 장소 검색을 제공하는 네이버의 지도 서비스입니다.' },
  { name: '당근', info: '지역 기반 중고거래와 동네 생활 정보를 공유하는 하이퍼로컬 커뮤니티 앱입니다.' },
  { name: '라인', info: '일본·동남아에서 널리 쓰이는 메신저로, 채팅과 함께 페이먼트·콘텐츠 서비스를 제공합니다.' },
  { name: '구글 크롬', info: '구글이 만든 빠르고 안정적인 웹 브라우저로, 다양한 확장 프로그램과 동기화 기능을 지원합니다.' },
  { name: '마이크로소프트 팀즈', info: '채팅·회의·파일 공유·협업을 통합 제공하는 마이크로소프트의 비즈니스 협업 도구입니다.' },
  { name: '왓츠앱', info: '전 세계에서 가장 많이 사용되는 메신저 중 하나로, 종단간 암호화 채팅과 통화를 제공합니다.' },
  { name: '텔레그램', info: '보안과 속도를 강조하는 메신저로, 대용량 파일 전송과 채널·봇 기능을 지원합니다.' }
];

window.loadApps = function () {
  try {
    var saved = localStorage.getItem(window.APP_STORAGE_KEY);
    if (saved) {
      var parsed = JSON.parse(saved);
      if (Array.isArray(parsed) && parsed.every(function (a) { return a && typeof a.name === 'string'; })) {
        return parsed.map(function (a) { return { name: a.name, info: String(a.info || ''), deleted: !!a.deleted }; });
      }
    }
  } catch (e) { /* ignore */ }
  return window.APP_INITIAL.slice();
};

window.saveApps = function (apps) {
  localStorage.setItem(window.APP_STORAGE_KEY, JSON.stringify(apps));
};

window.escapeHtml = function (s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c];
  });
};
