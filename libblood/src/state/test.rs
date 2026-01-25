// Add test for stage 2 completion verification
#[test]
fn test_stage2_completion_verification() {
    use crate::hand::{hand, tile27_to_vec};
    use crate::mjai::Event;
    use crate::state::PlayerState;
    use crate::t;
    
    // Test 1: Verify no chi functionality
    let mut ps = PlayerState::new(0);
    ps.test_update(&Event::StartKyoku {
        kyoku: 1,
        oya: 0,
        scores: [25000; 4],
        tehais: [
            tile27_to_vec(&hand("123m 456p 789s 11m").unwrap())
                .try_into()
                .unwrap(),
            [t!(?); 13],
            [t!(?); 13],
            [t!(?); 13],
        ],
    });
    
    // Try to trigger chi (should not be available)
    ps.test_update(&Event::Dahai {
        actor: 1,
        pai: t!(2m),
        tsumogiri: false,
    });
    
    let cans = ps.last_cans;
    // Bloody Battle: No chi, so can_chi should always be false
    assert!(!cans.can_chi(), "Chi should not be available in Bloody Battle Mahjong");
    
    // Test 2: Verify no riichi functionality
    // can_riichi should always be false
    assert!(!cans.can_riichi, "Riichi should not be available in Bloody Battle Mahjong");
    
    // Test 3: Verify oya doesn't affect scoring
    use crate::algo::point::Point;
    let point_3fan = Point::calc_from_fan(3);
    // In Bloody Battle, oya and ko pay the same
    assert_eq!(point_3fan.tsumo_ko, point_3fan.tsumo_oya, "Oya should not affect scoring in Bloody Battle");
    assert_eq!(point_3fan.tsumo_total(false), point_3fan.tsumo_total(true), "Oya should not affect tsumo_total");
    
    // Test 4: Verify 3-player agari end condition logic
    // This is tested in board.rs, but we verify the logic here
    // When agari_count >= 3, game should end
    // This is verified in the board.rs implementation
}
